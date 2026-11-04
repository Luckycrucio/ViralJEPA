"""exp04 JEPA pretraining of the exp01 CNN-Transformer on corpus_v3, with a CLS token inside the
encoder that gets its own sequence-level objectives. A frozen linear probe on ViraMiner tracks
progress, on the CLS output (what the classifier reads) and on the mean-pooled tokens.

Why exp03 failed: its variance hinge sat on the mean of the encoder's tokens, and context and
target saw the same window. A mean of 150 LayerNorm'd tokens only varies across the batch if the
tokens point the same way, so they collapsed onto one vector per sequence (token similarity
0.97), and the covariance penalty spread those vectors into a near-isotropic, input-sensitive
code (RankMe 285/288, targets of different sequences near-orthogonal: jepa_constant 0.89). The
probe fell from 0.80 to 0.58 AUROC, below a random encoder's composition-like features.

Changes from exp03:
  - a CLS token is prepended before the Transformer (Viraminer3), and the classifier reads it
    directly instead of adding its own CLS attention layer;
  - two views per step: window A, and window B cut from the same record (genome, contig or
    chromosome; `--views window` uses A again), each with its own RC/shift augmentation. Same-
    genome positives are what DNABERT-S uses; they reward what windows of one organism share
    (composition, codon usage), which is the signal viral detection rests on. No category or
    label is ever shown to the model, only which windows share a record;
  - CLS objective, BYOL-style: the masked A's CLS -> projector -> predictor must match the EMA
    target's projected CLS of the clean B (cosine);
  - the VICReg variance/covariance terms act on the CLS projector output (as in VICReg, on the
    expander, not on the representation) and no longer on pooled tokens, so tokens are free to
    differ from each other;
  - CLS composition target: the masked A's CLS predicts the 4-mer profile (both strands) of B
    (KL). Label-free, and aimed at what a random encoder already gets right;
  - token JEPA predicts every masked span token (I-JEPA) instead of the span mean, against
    data2vec targets: the mean of the target's last `--target-blocks` block outputs, each
    LayerNorm'd without affine parameters; MSE loss;
  - masking (spans + scattered MLM tokens) and the MLM objective are unchanged."""

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from model import (JEPAPredictor, ViralJEPAEncoder, augment, autocast, load_corpus, mlp, one_hot, parameter_count,
                   read_csv, reverse_complement)


def span_mask(batch_size, token_count, progress, device):
    """JEPA spans: 1-2 spans covering 10-20% of tokens at the start, rising linearly to 1-4 spans
    covering 20-30% at the end. Spans never overlap, so the coverage is exact."""
    low, high = 0.10 + 0.10 * progress, 0.20 + 0.10 * progress
    spans = torch.randint(1, 3 + round(2 * progress), (batch_size, 1), device=device)
    masked = ((low + (high - low) * torch.rand(batch_size, 1, device=device)) * token_count).round().long()
    slot = torch.arange(4, device=device)
    active = slot < spans
    length = torch.where(active, masked // spans + (slot < masked % spans).long(), 0)
    # Spread the unmasked tokens over the gaps before each span: sorted cut points in [0, unmasked].
    unmasked = token_count - masked
    cuts = torch.where(active, (torch.rand(batch_size, 4, device=device) * (unmasked + 1)).floor().long(), unmasked)
    starts = cuts.sort(1).values + length.cumsum(1) - length
    position = torch.arange(token_count, device=device)
    return ((position >= starts[..., None]) & (position < (starts + length)[..., None])).any(1)


def scattered_mask(span, rate):
    """MLM tokens: single tokens outside the JEPA spans, each with probability `rate`. Their
    neighbours stay visible, so the masked bases are predictable from local context."""
    return (torch.rand(span.shape, device=span.device) < rate) & ~span


def partner(index, group_start, group_size):
    """Another window of the same record when it has one, else the window itself. The corpus is
    sorted by group, so a group is the range [start, start + size)."""
    start, size = group_start[index], group_size[index]
    other = start + (torch.rand(len(index), device=index.device) * (size - 1)).long()
    other = other + (other >= index).long()  # skip the window itself
    return torch.where(size > 1, other, index)


def kmer_profile(codes, k=4):
    """Frequencies of the 4^k k-mers over both strands of each window; k-mers with N are skipped."""
    codes = codes.long()
    weights = 4 ** torch.arange(k - 1, -1, -1, device=codes.device)
    counts = torch.zeros(len(codes), 4 ** k, device=codes.device)
    for strand in (codes, reverse_complement(codes)):
        kmers = strand.unfold(1, k, 1)
        counts.scatter_add_(1, (kmers.clamp(max=3) * weights).sum(-1), (kmers < 4).all(-1).float())
    return counts / counts.sum(1, keepdim=True).clamp(min=1)


def variance_hinge(z, gamma):
    """VICReg variance: mean over dimensions of max(0, gamma - std across the batch)."""
    return F.relu(gamma - torch.sqrt(z.float().var(0) + 1e-4)).mean()


def covariance_penalty(z):
    """VICReg covariance: sum of squared off-diagonal covariances divided by d."""
    z = z.float() - z.float().mean(0)
    cov = z.T @ z / (len(z) - 1)
    return (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / z.shape[1]


@torch.no_grad()
def update_ema(online, target, momentum):
    online_params, target_params = list(online.parameters()), list(target.parameters())
    torch._foreach_mul_(target_params, momentum)
    torch._foreach_add_(target_params, online_params, alpha=1 - momentum)


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def embed(encoder, codes, device, batch_size=1024):
    """CLS and mean-pooled token features of clean windows, each averaged over both strands."""
    encoder.eval()
    cls_features, mean_features = [], []
    for start in range(0, len(codes), batch_size):
        batch = codes[start : start + batch_size].to(device)
        with autocast(device):
            forward = encoder(one_hot(batch))
            reverse = encoder(one_hot(reverse_complement(batch)))
        cls_features.append(((forward[0] + reverse[0]) / 2).float().cpu())
        mean_features.append(((forward[1].mean(1) + reverse[1].mean(1)) / 2).float().cpu())
    return torch.cat(cls_features).numpy(), torch.cat(mean_features).numpy()


def rankme(features):
    singular = np.linalg.svd(features - features.mean(0), compute_uv=False)
    p = singular / singular.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def fit_probe(train_x, train_y, evaluation_x, evaluation_y, prefix):
    scaler = StandardScaler().fit(train_x)
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(train_x), train_y)
    scores = probe.decision_function(scaler.transform(evaluation_x))
    return {
        f"{prefix}probe_auroc": float(roc_auc_score(evaluation_y, scores)),
        f"{prefix}probe_ap": float(average_precision_score(evaluation_y, scores)),
        f"{prefix}rankme": rankme(evaluation_x),
        f"{prefix}feature_std": float(evaluation_x.std(0).mean()),
    }


def linear_probe(encoder, train, evaluation, device):
    """Logistic regression on frozen features; AUROC/AP on the evaluation split, for the CLS output
    (unprefixed keys) and the mean-pooled tokens (`mean_` keys)."""
    train_cls, train_mean = embed(encoder, train[0], device)
    evaluation_cls, evaluation_mean = embed(encoder, evaluation[0], device)
    return {**fit_probe(train_cls, train[1], evaluation_cls, evaluation[1], ""),
            **fit_probe(train_mean, train[1], evaluation_mean, evaluation[1], "mean_")}


def save_plots(out_dir):
    history = pd.read_csv(out_dir / "history.csv")
    probes = pd.read_csv(out_dir / "probe_history.csv")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for column in ("jepa", "jepa_constant", "global", "global_constant", "kmer", "kmer_constant", "mlm",
                   "variance", "covariance"):
        axes[0].plot(history["step"], history[column], label=column)
    axes[0].set(title="Loss terms (unweighted)", xlabel="step", yscale="log")
    axes[0].legend(fontsize=7)
    axes[1].plot(history["step"], history["token_similarity"], label="token similarity")
    axes[1].plot(history["step"], history["mlm_accuracy"], label="MLM accuracy")
    axes[1].set(title="Token similarity / MLM accuracy", xlabel="step")
    axes[1].legend()
    axes[2].plot(probes["step"], probes["probe_ap"], marker="o", label="CLS val AP")
    axes[2].plot(probes["step"], probes["mean_probe_ap"], marker="o", label="mean-pool val AP")
    axes[2].axhline(probes["probe_ap"].iloc[0], color="gray", ls="--", label="random init (CLS)")
    axes[2].set(title="Frozen linear probe (fullset_validation)", xlabel="step")
    axes[2].legend()
    axes[3].plot(probes["step"], probes["rankme"], marker="o", label="CLS")
    axes[3].plot(probes["step"], probes["mean_rankme"], marker="o", label="mean-pool")
    axes[3].set(title="RankMe of features (max 288)", xlabel="step")
    axes[3].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pretrain_curves.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.save_dir) / "pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "encoder.pt").exists():
        print("Pretraining already finished:", out_dir / "encoder.pt")
        return
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    started = time.time()
    corpus, groups = load_corpus(args.corpus_dir, Path(args.save_dir) / "corpus_codes.npy")
    if args.max_windows:
        corpus, groups = corpus[: args.max_windows], groups[: args.max_windows]
    # Sort by group, so that each group is a contiguous range of rows.
    order = np.argsort(groups, kind="stable")
    corpus, groups = corpus[order], groups[order]
    _, first, inverse, size = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
    group_start = torch.from_numpy(first[inverse]).to(device)
    group_size = torch.from_numpy(size[inverse]).to(device)
    corpus = torch.from_numpy(corpus).to(device)
    print(f"Corpus: {len(corpus):,} windows in {len(size):,} records ({int((size == 1).sum()):,} single-window), "
          f"loaded in {time.time() - started:.0f} s")

    data = Path(args.data_path)
    _, train_codes, train_labels = read_csv(data / "fullset_train.csv")
    _, val_codes, val_labels = read_csv(data / "fullset_validation.csv")
    _, test_codes, test_labels = read_csv(data / "fullset_test.csv")
    subset = rng.choice(len(train_codes), min(args.probe_train_size, len(train_codes)), replace=False)
    probe_train = (torch.from_numpy(train_codes[subset]), train_labels[subset])
    probe_val = (torch.from_numpy(val_codes), val_labels)

    dim = 288
    encoder = ViralJEPAEncoder(transformer_dim=dim).to(device)
    projector = mlp(dim, args.projector_dim, args.projector_dim).to(device)
    target = deepcopy(nn.ModuleList([encoder, projector])).requires_grad_(False).eval()
    target_encoder, target_projector = target
    predictor = JEPAPredictor(dim, encoder.token_count).to(device)
    global_predictor = mlp(args.projector_dim, args.projector_dim, args.projector_dim).to(device)
    kmer_head = nn.Linear(dim, 4 ** 4).to(device)
    mlm_head = nn.Linear(dim, 2 * 4).to(device)  # 2 bases per token (stem stride 2)
    online = nn.ModuleList([encoder, predictor, mlm_head, projector, global_predictor, kmer_head])
    no_decay_names = ("pos", "mask_token", "cls_token")
    decay = [p for n, p in online.named_parameters() if p.ndim >= 2 and not any(s in n for s in no_decay_names)]
    no_decay = [p for n, p in online.named_parameters() if not (p.ndim >= 2 and not any(s in n for s in no_decay_names))]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    print(f"Encoder {parameter_count(encoder):,} | predictor {parameter_count(predictor):,} | "
          f"projector + global predictor {parameter_count(projector) + parameter_count(global_predictor):,} parameters")

    checkpoint_path = out_dir / "checkpoint.pt"
    step, history, probes = 0, [], []
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        online.load_state_dict(state["online"])
        target.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        # The CSVs are written more often than the checkpoint: drop rows logged after it.
        history = [row for row in pd.read_csv(out_dir / "history.csv").to_dict("records") if row["step"] <= step]
        probes = [row for row in pd.read_csv(out_dir / "probe_history.csv").to_dict("records") if row["step"] <= step]
        print(f"Resumed at step {step}")
    else:
        probes.append({"step": 0, **linear_probe(target_encoder, probe_train, probe_val, device)})
        pd.DataFrame(probes).to_csv(out_dir / "probe_history.csv", index=False)
        print("Random init probe:", probes[-1])

    token_count = encoder.token_count
    order = torch.randperm(len(corpus), device=device)
    cursor = (step * args.batch_size) % len(corpus)
    running, window_started = [], time.time()

    while step < args.steps:
        if cursor + args.batch_size > len(corpus):
            order, cursor = torch.randperm(len(corpus), device=device), 0
        index = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        other = partner(index, group_start, group_size) if args.views == "record" else index
        view_a, view_b = augment(corpus[index]), augment(corpus[other])
        progress = step / args.steps
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        momentum = args.ema_start + (args.ema_end - args.ema_start) * progress

        span = span_mask(len(view_a), token_count, progress, device)
        mlm_tokens = scattered_mask(span, args.mlm_rate)
        token_mask = span | mlm_tokens
        with autocast(device):
            # Hidden bases become N before the stem (no visible token's convolution sees them) and hidden
            # tokens become the mask token after it.
            context_cls, context = encoder(one_hot(view_a.masked_fill(token_mask.repeat_interleave(2, dim=1), 4)),
                                           token_mask)
            with torch.no_grad():
                _, target_tokens, blocks = target_encoder(one_hot(view_a), return_blocks=args.target_blocks)
                tokens_target = torch.stack([F.layer_norm(b.float(), b.shape[-1:]) for b in blocks]).mean(0)
                target_z = target_projector(target_encoder(one_hot(view_b))[0]).float()
                composition = kmer_profile(view_b)

            # Token JEPA: every masked span token, against the data2vec target. jepa_constant is the loss
            # of predicting the batch-mean target at each position.
            predicted = predictor(context_cls, context, token_mask).float()
            jepa = (predicted - tokens_target).pow(2).mean(-1)[span].mean()
            jepa_constant = (tokens_target - tokens_target.mean(0, keepdim=True)).pow(2).mean(-1)[span].mean()

            # CLS: BYOL-style prediction of the other view's target CLS, VICReg on the projector output,
            # and the other view's 4-mer profile.
            z = projector(context_cls).float()
            global_loss = 1 - F.cosine_similarity(global_predictor(z).float(), target_z, dim=-1).mean()
            global_constant = 1 - F.cosine_similarity(target_z.mean(0, keepdim=True), target_z, dim=-1).mean()
            variance, covariance = variance_hinge(z, args.var_gamma), covariance_penalty(z)
            kmer = F.kl_div(F.log_softmax(kmer_head(context_cls).float(), -1), composition, reduction="batchmean")
            kmer_constant = F.kl_div(composition.mean(0, keepdim=True).clamp(min=1e-8).log().expand_as(composition),
                                     composition, reduction="batchmean")

            logits = mlm_head(context).float().view(len(view_a), -1, 4)
            bases = view_a.long().masked_fill(~mlm_tokens.repeat_interleave(2, dim=1) | (view_a == 4), -100)
            mlm = F.cross_entropy(logits.reshape(-1, 4), bases.reshape(-1), ignore_index=-100)

            loss = (args.jepa_weight * jepa + args.global_weight * global_loss + args.kmer_weight * kmer
                    + args.mlm_weight * mlm + args.var_weight * variance + args.cov_weight * covariance)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.clip)
        optimizer.step()
        update_ema(nn.ModuleList([encoder, projector]), target, momentum)
        step += 1

        with torch.no_grad():
            accuracy = (logits.argmax(-1) == bases)[bases >= 0].float().mean()
            # Cosine between each target token and its sequence's mean: near 1 means the tokens of a
            # sequence have collapsed onto one vector (exp03: 0.97).
            target_tokens = target_tokens.float()
            token_similarity = F.cosine_similarity(target_tokens, target_tokens.mean(1, keepdim=True), dim=-1).mean()
        running.append(torch.stack([loss.detach(), jepa.detach(), jepa_constant, global_loss.detach(), global_constant,
                                    kmer.detach(), kmer_constant, mlm.detach(), accuracy, variance.detach(),
                                    covariance.detach(), context_cls.float().std(0).mean().detach(),
                                    token_similarity, token_mask.float().mean(), (other != index).float().mean()]))

        if step % args.log_every == 0 or step == args.steps:
            values = torch.stack(running).mean(0).tolist()
            elapsed = time.time() - window_started
            history.append({
                "step": step, "epoch": step * args.batch_size / len(corpus), "lr": optimizer.param_groups[0]["lr"],
                "ema_momentum": momentum, **dict(zip(("loss", "jepa", "jepa_constant", "global", "global_constant", "kmer",
                                                      "kmer_constant", "mlm", "mlm_accuracy", "variance", "covariance",
                                                      "cls_std", "token_similarity", "mask_ratio", "distinct_views"), values)),
                "sequences_per_second": len(running) * args.batch_size / elapsed,
                "hours": (history[-1]["hours"] if history else 0.0) + elapsed / 3600,
            })
            running, window_started = [], time.time()
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
            row = history[-1]
            print(f"step {step:6d} | loss {row['loss']:.4f} jepa {row['jepa']:.4f} (const {row['jepa_constant']:.4f}) "
                  f"global {row['global']:.4f} (const {row['global_constant']:.4f}) "
                  f"kmer {row['kmer']:.4f} (const {row['kmer_constant']:.4f}) mlm {row['mlm']:.3f} "
                  f"acc {row['mlm_accuracy']:.3f} var {row['variance']:.3f} cov {row['covariance']:.4f} "
                  f"cls std {row['cls_std']:.3f} tsim {row['token_similarity']:.3f} mask {row['mask_ratio']:.2f} | "
                  f"{row['sequences_per_second']:.0f} seq/s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            probes.append({"step": step, **linear_probe(target_encoder, probe_train, probe_val, device)})
            pd.DataFrame(probes).to_csv(out_dir / "probe_history.csv", index=False)
            print("Probe:", probes[-1], flush=True)
            torch.save({"online": online.state_dict(), "target": target.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step, "args": vars(args)}, checkpoint_path)

    # Final gate: probe on the full fullset_train, scored on fullset_test, JEPA vs random init.
    full_train = (torch.from_numpy(train_codes), train_labels)
    test = (torch.from_numpy(test_codes), test_labels)
    torch.manual_seed(args.seed)
    gate = {
        "random_init": linear_probe(ViralJEPAEncoder(transformer_dim=dim).to(device), full_train, test, device),
        "jepa_target_encoder": linear_probe(target_encoder, full_train, test, device),
        "steps": step,
        "pretrain_hours": history[-1]["hours"] if history else None,
    }
    (out_dir / "probe_test.json").write_text(json.dumps(gate, indent=2) + "\n")
    print("Test probe:", gate)
    torch.save({"target_encoder": target_encoder.state_dict(), "context_encoder": encoder.state_dict(),
                "args": vars(args)}, out_dir / "encoder.pt")
    save_plots(out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="exp04 JEPA pretraining with a CLS token on corpus_v3")
    parser.add_argument("--corpus-dir", required=True, help="folder with pretrain_corpus_*.csv")
    parser.add_argument("--data-path", required=True, help="ViraMiner dataset folder (probe only)")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=3_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--views", choices=["record", "window"], default="record",
                        help="second view: another window of the same record, or the same window re-augmented")
    parser.add_argument("--target-blocks", type=int, default=3, help="token targets: mean of the last K blocks")
    parser.add_argument("--projector-dim", type=int, default=1024)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--kmer-weight", type=float, default=1.0, help="0 turns off the CLS 4-mer target")
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--mlm-rate", type=float, default=0.15, help="MLM masking rate of tokens outside the JEPA spans")
    parser.add_argument("--var-weight", type=float, default=1.0)
    parser.add_argument("--var-gamma", type=float, default=1.0, help="variance hinge target on the CLS projector output")
    parser.add_argument("--cov-weight", type=float, default=0.04)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--probe-train-size", type=int, default=50_000)
    parser.add_argument("--max-windows", type=int, default=0, help="debug: use only the first N windows")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

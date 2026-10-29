"""exp02 JEPA pretraining of the exp01 CNN-Transformer on corpus_v3, following JEPA-DNA
(arXiv 2602.17162): sequence-level JEPA (cosine) + token-level MLM + VICReg variance/covariance,
scheduled span masking, EMA target encoder. A frozen linear probe on ViraMiner tracks progress."""

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

from model import (JEPAPredictor, ViralJEPAEncoder, augment, autocast, load_corpus, one_hot, parameter_count, read_csv,
                   reverse_complement)


def span_mask(batch_size, token_count, progress, device):
    """JEPA-DNA schedule: 1-2 spans covering 10-30% of tokens at the start, rising linearly to
    1-4 spans covering 30-50% at the end. Spans may overlap."""
    low, high = 0.10 + 0.20 * progress, 0.30 + 0.20 * progress
    spans = torch.randint(1, 3 + round(2 * progress), (batch_size, 1), device=device)
    ratio = low + (high - low) * torch.rand(batch_size, 1, device=device)
    length = (ratio * token_count / spans).round().clamp(min=1)
    starts = (torch.rand(batch_size, 4, device=device) * (token_count - length + 1)).floor()
    position = torch.arange(token_count, device=device)
    inside = (position >= starts[..., None]) & (position < (starts + length)[..., None])
    active = torch.arange(4, device=device) < spans
    return (inside & active[..., None]).any(1)


def vicreg(z, gamma=1.0):
    z = z.float()
    std = torch.sqrt(z.var(0) + 1e-4)
    centered = z - z.mean(0)
    cov = centered.T @ centered / (len(z) - 1)
    off_diagonal = cov - torch.diag(torch.diag(cov))
    return F.relu(gamma - std).mean(), off_diagonal.pow(2).sum() / z.shape[1]


@torch.no_grad()
def update_ema(online, target, momentum):
    online_params, target_params = list(online.parameters()), list(target.parameters())
    torch._foreach_mul_(target_params, momentum)
    torch._foreach_add_(target_params, online_params, alpha=1 - momentum)
    for online_buffer, target_buffer in zip(online.buffers(), target.buffers()):
        target_buffer.copy_(online_buffer)


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def embed(encoder, codes, device, batch_size=1024):
    """Mean-pooled clean token features, averaged over both strands."""
    encoder.eval()
    features = []
    for start in range(0, len(codes), batch_size):
        batch = codes[start : start + batch_size].to(device)
        with autocast(device):
            forward = encoder(one_hot(batch)).mean(1)
            reverse = encoder(one_hot(reverse_complement(batch))).mean(1)
        features.append(((forward + reverse) / 2).float().cpu())
    return torch.cat(features).numpy()


def rankme(features):
    singular = np.linalg.svd(features - features.mean(0), compute_uv=False)
    p = singular / singular.sum() + 1e-12
    return float(np.exp(-(p * np.log(p)).sum()))


def linear_probe(encoder, train, evaluation, device):
    """Logistic regression on frozen features; returns AUROC/AP on the evaluation split."""
    train_x, evaluation_x = embed(encoder, train[0], device), embed(encoder, evaluation[0], device)
    scaler = StandardScaler().fit(train_x)
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(train_x), train[1])
    scores = probe.decision_function(scaler.transform(evaluation_x))
    return {
        "probe_auroc": float(roc_auc_score(evaluation[1], scores)),
        "probe_ap": float(average_precision_score(evaluation[1], scores)),
        "rankme": rankme(evaluation_x),
        "feature_std": float(evaluation_x.std(0).mean()),
    }


def save_plots(out_dir):
    history = pd.read_csv(out_dir / "history.csv")
    probes = pd.read_csv(out_dir / "probe_history.csv")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for column in ("jepa", "mlm", "variance", "covariance"):
        axes[0].plot(history["step"], history[column], label=column)
    axes[0].set(title="Loss terms (unweighted)", xlabel="step", yscale="log")
    axes[0].legend()
    axes[1].plot(history["step"], history["mlm_accuracy"])
    axes[1].set(title="MLM accuracy on masked bases", xlabel="step")
    axes[2].plot(probes["step"], probes["probe_ap"], marker="o", label="val AP")
    axes[2].axhline(probes["probe_ap"].iloc[0], color="gray", ls="--", label="random init")
    axes[2].set(title="Frozen linear probe (fullset_validation)", xlabel="step")
    axes[2].legend()
    axes[3].plot(probes["step"], probes["rankme"], marker="o")
    axes[3].set(title="RankMe of pooled features (max 288)", xlabel="step")
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
    corpus = torch.from_numpy(load_corpus(args.corpus_dir, Path(args.save_dir) / "corpus_codes.npy"))
    if args.max_windows:
        corpus = corpus[: args.max_windows]
    corpus = corpus.to(device)
    print(f"Corpus: {len(corpus):,} windows loaded in {time.time() - started:.0f} s")

    data = Path(args.data_path)
    _, train_codes, train_labels = read_csv(data / "fullset_train.csv")
    _, val_codes, val_labels = read_csv(data / "fullset_validation.csv")
    _, test_codes, test_labels = read_csv(data / "fullset_test.csv")
    subset = rng.choice(len(train_codes), min(args.probe_train_size, len(train_codes)), replace=False)
    probe_train = (torch.from_numpy(train_codes[subset]), train_labels[subset])
    probe_val = (torch.from_numpy(val_codes), val_labels)

    encoder = ViralJEPAEncoder().to(device)
    target_encoder = deepcopy(encoder).requires_grad_(False).eval()
    predictor = JEPAPredictor(encoder.transformer_dim, encoder.token_count).to(device)
    mlm_head = nn.Linear(encoder.transformer_dim, 2 * 4).to(device)  # 2 bases per token (stem stride 2)
    online = nn.ModuleList([encoder, predictor, mlm_head])
    decay = [p for n, p in online.named_parameters() if p.ndim >= 2 and "pos" not in n and "mask_token" not in n]
    no_decay = [p for n, p in online.named_parameters() if not (p.ndim >= 2 and "pos" not in n and "mask_token" not in n)]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    print(f"Encoder {parameter_count(encoder):,} | predictor {parameter_count(predictor):,} parameters")

    checkpoint_path = out_dir / "checkpoint.pt"
    step, history, probes = 0, [], []
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        online.load_state_dict(state["online"])
        target_encoder.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        history = pd.read_csv(out_dir / "history.csv").to_dict("records")
        probes = pd.read_csv(out_dir / "probe_history.csv").to_dict("records")
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
        codes = augment(corpus[order[cursor : cursor + args.batch_size]])
        cursor += args.batch_size
        progress = step / args.steps
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        momentum = args.ema_start + (args.ema_end - args.ema_start) * progress

        token_mask = span_mask(len(codes), token_count, progress, device)
        base_mask = token_mask.repeat_interleave(2, dim=1)
        with autocast(device):
            context = encoder(one_hot(codes.masked_fill(base_mask, 4)))
            predicted = predictor(context, token_mask)
            with torch.no_grad():
                target = target_encoder(one_hot(codes)).mean(1)
            jepa = 1 - F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()

            logits = mlm_head(context).float().view(len(codes), -1, 4)
            bases = codes.long().masked_fill(~base_mask | (codes == 4), -100)
            mlm = F.cross_entropy(logits.reshape(-1, 4), bases.reshape(-1), ignore_index=-100)

            # Variance/covariance on the clean, unmasked sequence representation (eval mode).
            encoder.eval(), predictor.eval()
            clean = predictor(encoder(one_hot(codes)))
            encoder.train(), predictor.train()
            variance, covariance = vicreg(clean)
            loss = args.mlm_weight * mlm + args.jepa_weight * jepa + args.var_weight * variance + args.cov_weight * covariance

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.clip)
        optimizer.step()
        update_ema(encoder, target_encoder, momentum)
        step += 1

        with torch.no_grad():
            accuracy = (logits.argmax(-1) == bases)[bases >= 0].float().mean()
        running.append(torch.stack([loss.detach(), jepa.detach(), mlm.detach(), accuracy, variance.detach(),
                                    covariance.detach(), clean.float().std(0).mean().detach(),
                                    token_mask.float().mean()]))

        if step % args.log_every == 0 or step == args.steps:
            values = torch.stack(running).mean(0).tolist()
            elapsed = time.time() - window_started
            history.append({
                "step": step, "epoch": step * args.batch_size / len(corpus), "lr": optimizer.param_groups[0]["lr"],
                "ema_momentum": momentum, **dict(zip(("loss", "jepa", "mlm", "mlm_accuracy", "variance",
                                                      "covariance", "representation_std", "mask_ratio"), values)),
                "sequences_per_second": len(running) * args.batch_size / elapsed,
                "hours": (history[-1]["hours"] if history else 0.0) + elapsed / 3600,
            })
            running, window_started = [], time.time()
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
            row = history[-1]
            print(f"step {step:6d} | loss {row['loss']:.4f} jepa {row['jepa']:.4f} mlm {row['mlm']:.3f} "
                  f"acc {row['mlm_accuracy']:.3f} var {row['variance']:.3f} cov {row['covariance']:.4f} "
                  f"mask {row['mask_ratio']:.2f} | {row['sequences_per_second']:.0f} seq/s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            probes.append({"step": step, **linear_probe(target_encoder, probe_train, probe_val, device)})
            pd.DataFrame(probes).to_csv(out_dir / "probe_history.csv", index=False)
            print("Probe:", probes[-1], flush=True)
            torch.save({"online": online.state_dict(), "target": target_encoder.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step, "args": vars(args)}, checkpoint_path)

    # Final gate: probe on the full fullset_train, scored on fullset_test, JEPA vs random init.
    full_train = (torch.from_numpy(train_codes), train_labels)
    test = (torch.from_numpy(test_codes), test_labels)
    torch.manual_seed(args.seed)
    gate = {
        "random_init": linear_probe(ViralJEPAEncoder().to(device), full_train, test, device),
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
    parser = argparse.ArgumentParser(description="exp02 JEPA-DNA style pretraining on corpus_v3")
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
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--mlm-weight", type=float, default=0.1)
    parser.add_argument("--var-weight", type=float, default=25.0)
    parser.add_argument("--cov-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--probe-train-size", type=int, default=50_000)
    parser.add_argument("--max-windows", type=int, default=0, help="debug: use only the first N windows")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""exp02 supervised fine-tuning with the exp01 recipe (CLS head, focal loss, AdamW, 120 epochs,
patience 15). Two tasks:
  fullset  train fullset_train, select on fullset_validation, test fullset_test
  anell    train training_data_anell (exact test duplicates removed, 10% stratified validation),
           test test_data_anell: anelloviruses are absent from both fine-tuning and corpus_v3
Model selection uses validation AP. Every evaluation reports plain and reverse-complement
averaged (RC-TTA) scores, with bootstrap 95% CIs on test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score

from model import ViralJEPAClassifier, ViralJEPAEncoder, augment, autocast, one_hot, parameter_count, read_csv, reverse_complement


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, targets):
        targets = targets[:, None] * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (self.alpha * (1 - torch.exp(-bce)) ** self.gamma * bce).mean()


def load_task(data, task, seed=0):
    """(train, validation, test) splits as (ids, codes, labels) tuples."""
    if task == "fullset":
        return tuple(read_csv(data / f"fullset_{name}.csv") for name in ("train", "validation", "test"))
    ids, codes, labels = read_csv(data / "training_data_anell_300_binary.csv")
    test = read_csv(data / "test_data_anell_300_binary.csv")
    test_sequences = {row.tobytes() for row in test[1]}
    keep = np.array([row.tobytes() not in test_sequences for row in codes])
    print(f"anell: dropped {int((~keep).sum())} training windows identical to a test window")
    ids, codes, labels = ids[keep], codes[keep], labels[keep]
    rng = np.random.default_rng(seed)
    validation = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        validation[rng.choice(members, round(0.1 * len(members)), replace=False)] = True
    return (ids[~validation], codes[~validation], labels[~validation]), (ids[validation], codes[validation], labels[validation]), test


@torch.no_grad()
def predict(model, codes, device, batch_size=1024):
    """Plain and RC-averaged virus probabilities."""
    model.eval()
    plain, averaged = [], []
    for start in range(0, len(codes), batch_size):
        batch = codes[start : start + batch_size].to(device)
        with autocast(device):
            forward = torch.sigmoid(model(one_hot(batch)).float())
            reverse = torch.sigmoid(model(one_hot(reverse_complement(batch))).float())
        plain.append(forward.cpu())
        averaged.append(((forward + reverse) / 2).cpu())
    return torch.cat(plain).flatten().numpy(), torch.cat(averaged).flatten().numpy()


def f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def bootstrap(labels, probabilities, samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    auroc, ap = [], []
    for _ in range(samples):
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() == labels[index].max():
            continue
        auroc.append(roc_auc_score(labels[index], probabilities[index]))
        ap.append(average_precision_score(labels[index], probabilities[index]))
    return {"auroc_ci95": np.percentile(auroc, [2.5, 97.5]).tolist(), "ap_ci95": np.percentile(ap, [2.5, 97.5]).tolist()}


def metrics(labels, probabilities, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, (probabilities >= threshold).astype(int), labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "ap": float(average_precision_score(labels, probabilities)),
        "threshold": threshold,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        **bootstrap(labels, probabilities),
    }


def main():
    args = parse_args()
    out_dir = Path(args.save_dir) / "finetune" / f"{args.task}_{args.init}_s{args.seed}"
    if (out_dir / "result.json").exists():
        print("Already done:", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, validation, test = load_task(Path(args.data_path), args.task)
    train_codes, train_labels = torch.from_numpy(train[1]), torch.from_numpy(train[2]).float()
    validation_codes, test_codes = torch.from_numpy(validation[1]), torch.from_numpy(test[1])
    print(f"{args.task}: train {len(train_labels):,} ({int(train_labels.sum())} viral) | "
          f"validation {len(validation[2]):,} ({int(validation[2].sum())}) | test {len(test[2]):,} ({int(test[2].sum())})")

    encoder = ViralJEPAEncoder()
    if args.init == "jepa":
        encoder.load_state_dict(torch.load(args.encoder, map_location="cpu")["target_encoder"])
    model = ViralJEPAClassifier(encoder).to(device)

    transformer = [p for n, p in model.named_parameters() if n.startswith(("encoder.transformer.", "cls_attention."))]
    other = [p for n, p in model.named_parameters() if not n.startswith(("encoder.transformer.", "cls_attention."))]
    optimizer = torch.optim.AdamW(
        [{"params": other, "lr": args.lr}, {"params": transformer, "lr": args.transformer_lr}],
        weight_decay=args.weight_decay,
    )
    schedule = lambda epoch: (epoch + 1) / args.warmup_epochs if epoch < args.warmup_epochs else 0.5 * (
        1 + np.cos(np.pi * (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    criterion = FocalLoss()

    best_ap, best_epoch, history, started = -1.0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        order = torch.randperm(len(train_codes), generator=generator)
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            codes, labels = augment(train_codes[index].to(device)), train_labels[index].to(device)
            with autocast(device):
                loss = criterion(model(one_hot(codes)).float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total, count = total + loss.item() * len(index), count + len(index)
        scheduler.step()

        plain, averaged = predict(model, validation_codes, device)
        row = {
            "epoch": epoch, "train_loss": total / count,
            "val_auroc": roc_auc_score(validation[2], plain), "val_ap": average_precision_score(validation[2], plain),
            "val_auroc_rc": roc_auc_score(validation[2], averaged), "val_ap_rc": average_precision_score(validation[2], averaged),
            "lr": optimizer.param_groups[0]["lr"], "minutes": (time.time() - started) / 60,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if row["val_ap_rc"] > best_ap:
            best_ap, best_epoch = row["val_ap_rc"], epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        print(f"epoch {epoch:3d} | loss {row['train_loss']:.4f} | val AUROC {row['val_auroc_rc']:.4f} "
              f"AP {row['val_ap_rc']:.4f} (RC-TTA) | best AP {best_ap:.4f} @ {best_epoch}", flush=True)
        if epoch - best_epoch >= args.patience:
            break

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    validation_plain, validation_averaged = predict(model, validation_codes, device)
    test_plain, test_averaged = predict(model, test_codes, device)
    result = {
        "task": args.task, "init": args.init, "seed": args.seed,
        "encoder": args.encoder if args.init == "jepa" else None,
        "parameters": parameter_count(model),
        "best_epoch": best_epoch, "epochs_run": len(history), "minutes": (time.time() - started) / 60,
        "validation_ap_rc": best_ap,
        "test": metrics(test[2], test_plain, f1_threshold(validation[2], validation_plain)),
        "test_rc": metrics(test[2], test_averaged, f1_threshold(validation[2], validation_averaged)),
    }
    pd.DataFrame({"id": test[0], "label": test[2], "probability": test_plain, "probability_rc": test_averaged}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="exp02 fine-tuning on ViraMiner")
    parser.add_argument("--task", choices=["fullset", "anell"], required=True)
    parser.add_argument("--init", choices=["jepa", "random"], required=True)
    parser.add_argument("--encoder", help="pretrain/encoder.pt (for --init jepa)")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--transformer-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    main()

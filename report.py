"""Collect exp02 results into report.md (mean ± std over seeds) and report.png."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# exp01 (token-level JEPA on fullset_train only), single seed, fullset_test.
EXP01 = {"auroc": 0.9145, "ap": 0.4639, "auroc_rc": 0.9224, "ap_rc": 0.508}


def mean_std(values):
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    root = Path(parser.parse_args().save_dir)
    lines = ["# exp02 report", ""]

    pretrain = root / "pretrain"
    if (pretrain / "probe_test.json").exists():
        gate = json.loads((pretrain / "probe_test.json").read_text())
        history = pd.read_csv(pretrain / "history.csv").iloc[-1]
        lines += [
            "## Pretraining",
            "",
            f"{gate['steps']:,} steps, {gate['pretrain_hours']:.1f} h, last window: jepa {history['jepa']:.4f}, "
            f"mlm {history['mlm']:.3f} (acc {history['mlm_accuracy']:.3f}), var {history['variance']:.3f}, "
            f"cov {history['covariance']:.4f}, mask {history['mask_ratio']:.2f}.",
            "",
            "Frozen linear probe (train fullset_train, score fullset_test, RC-averaged features):",
            "",
            "| encoder | AUROC | AP | RankMe |",
            "|---|---|---|---|",
            *(f"| {name} | {p['probe_auroc']:.4f} | {p['probe_ap']:.4f} | {p['rankme']:.1f} |"
              for name, p in (("random init", gate["random_init"]), ("JEPA (EMA target)", gate["jepa_target_encoder"]))),
            "",
        ]

    rows = [json.loads(path.read_text()) for path in sorted((root / "finetune").glob("*/result.json"))]
    if rows:
        lines += [
            "## Fine-tuning (test set, mean ± std over seeds)",
            "",
            "| task | init | seeds | AUROC | AP | AUROC RC-TTA | AP RC-TTA | AP 95% CI (RC, seed 0) | F1 RC | precision RC | recall RC |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        if any(r["task"] == "fullset" for r in rows):
            lines.append(f"| fullset | exp01 | 1 | {EXP01['auroc']:.4f} | {EXP01['ap']:.4f} | {EXP01['auroc_rc']:.4f} | {EXP01['ap_rc']:.4f} | | | | |")
        for (task, init), group in pd.DataFrame(rows).groupby(["task", "init"]):
            group = group.sort_values("seed")
            plain, rc = list(group["test"]), list(group["test_rc"])
            ci = rc[0]["ap_ci95"]
            lines.append(
                f"| {task} | {init} | {len(group)} | {mean_std([m['auroc'] for m in plain])} | {mean_std([m['ap'] for m in plain])} | "
                f"{mean_std([m['auroc'] for m in rc])} | {mean_std([m['ap'] for m in rc])} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                f"{mean_std([m['f1'] for m in rc])} | {mean_std([m['precision'] for m in rc])} | {mean_std([m['recall'] for m in rc])} |"
            )
        lines += ["", "fullset = fullset_test (551 viral / 26,405). anell = test_data_anell (1,348 viral / 27,644), "
                  "anelloviruses unseen in pretraining and fine-tuning. Threshold = best F1 on validation.", ""]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, task in zip(axes, ("fullset", "anell")):
            for path in sorted((root / "finetune").glob(f"{task}_*/history.csv")):
                history = pd.read_csv(path)
                axis.plot(history["epoch"], history["val_ap_rc"], label=path.parent.name,
                          ls="-" if "_jepa_" in path.parent.name else "--")
            axis.set(title=f"{task}: validation AP (RC-TTA)", xlabel="epoch")
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "report.png", dpi=150)
        plt.close(fig)

    (root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

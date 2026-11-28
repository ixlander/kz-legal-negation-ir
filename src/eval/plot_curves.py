"""Learning curves of the NLI fine-tuning run logged with --curve-every.

Left: training loss (mean over 50 optimizer steps) and validation loss on the selection half of
Kardeş-NLU kk. Right: validation macro F1 and the learning-rate schedule.

    python -m src.eval.plot_curves --run a-curves
Writes results/figures/nli_learning_curves.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="a-curves")
    a = ap.parse_args()
    c = json.loads((ROOT / f"results/nli_curves_{a.run}.json").read_text(encoding="utf-8"))
    tr, va = c["train"], c["val"]
    per_epoch = c["steps_total"] / c["config"]["epochs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([p["step"] for p in tr], [p["loss"] for p in tr], color="#9aa5b1", lw=1, label="train loss (50-step mean)")
    ax1.plot([p["step"] for p in va], [p["loss"] for p in va], "o-", color="#c2410c", ms=3, label="validation loss")
    ax1.set_xlabel("optimizer step (effective batch 32)")
    ax1.set_ylabel("cross-entropy")
    ax1.legend()
    ax2.plot([p["step"] for p in va], [p["macro_f1"] for p in va], "o-", color="#1d4ed8", ms=3,
             label="validation macro F1")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("macro F1")
    lr = ax2.twinx()
    lr.plot([p["step"] for p in tr], [p["lr"] for p in tr], color="#9aa5b1", lw=1, ls="--", label="learning rate")
    lr.set_ylabel("learning rate")
    for ax in (ax1, ax2):
        ax.axvline(per_epoch, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.3)
    ax2.legend(loc="lower right")
    cfg = c["config"]
    fig.suptitle(f"mDeBERTa-v3-base NLI fine-tuning, variant {cfg['variant']}: {cfg['n_train']:,} pairs, "
                 f"AdamW lr {cfg['lr']:g}, batch {cfg['batch_size']}, seed {cfg['seed']}; "
                 f"validation = Kardeş-NLU kk selection half", fontsize=9)
    fig.tight_layout()
    out = ROOT / "results/figures/nli_learning_curves.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()

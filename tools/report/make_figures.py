"""Figures for the report page; reads results/ only.

Usage:
    python tools/report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"ru": "#2a78d6", "kz": "#eb6834"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
NAMES = {"e5-base": "multilingual-e5-base", "e5-large-instruct": "multilingual-e5-large-instruct",
         "bge-m3": "bge-m3", "labse": "LaBSE", "qwen3-emb-0.6b": "Qwen3-Embedding-0.6B",
         "embeddinggemma-2": "embeddinggemma-2"}


def d1_pool():
    ru = json.load(open(R / "d1_pool_ru.json", encoding="utf-8"))["models"]
    kz = json.load(open(R / "d1_pool_kz_prelim.json", encoding="utf-8"))["models"]
    keys = sorted(NAMES, key=lambda k: -(ru[k]["negation_position"] + kz[k]["negation_position"]) / 2)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, k in enumerate(keys):
        a, b = ru[k]["negation_position"], kz[k]["negation_position"]
        # Ru slightly above, Kz slightly below the row, so coinciding values stay visible
        ax.plot([a, b], [i - 0.13, i + 0.13], color=GRID, lw=2, zorder=1)
        for lang, v, dy in (("ru", a, -0.13), ("kz", b, 0.13)):
            ax.plot(v, i + dy, "o", ms=8, color=COL[lang], mec=SURF, mew=1.5, zorder=2,
                    label={"ru": "Russian", "kz": "Kazakh (prelim)"}[lang] if i == 0 else None)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([NAMES[k] for k in keys], color=INK)
    ax.set_ylim(len(keys) - 0.5, -0.9)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.axvline(1, color=INK2, lw=1, ls=":")
    box = dict(boxstyle="square,pad=0.15", facecolor=SURF, edgecolor="none")
    ax.text(0.03, -0.6, "0 = moved like a synonym edit", fontsize=8, color=INK2, va="center", bbox=box, zorder=3)
    ax.text(0.97, -0.6, "1 = moved like a rewording", fontsize=8, color=INK2, va="center", ha="right", bbox=box,
            zorder=3)
    ax.set_xlim(-0.1, 1.25)
    ax.set_xlabel("negation position (median over 30 norms)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("D1 across six embedders: where a negation lands", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(R / "d1_pool_position.png", dpi=160)


def d2_pool():
    d = json.load(open(R / "d2_pool.json", encoding="utf-8"))["tokenizers"]
    items = sorted(((k, v["kz"]["isolation_rate"] * 100) for k, v in d.items() if "kz" in v), key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, (k, v) in enumerate(items):
        is_xlmr = k.startswith("xlm-r")
        ax.barh(i, v, height=0.62, color="#2a78d6" if is_xlmr else "#9fb9dc", edgecolor=SURF, lw=2)
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], color=INK, fontsize=9)
    ax.set_xlim(0, 60)
    ax.set_xlabel("% of 154 Kazakh negative forms with the affix as its own token")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("D2 across 14 tokenizers (XLM-R = e5, bge-m3)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(R / "d2_pool_isolation.png", dpi=160)


if __name__ == "__main__":
    d1_pool()
    d2_pool()
    print("ok")

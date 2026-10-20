"""D1 — Negation Margin.

margin = cos(s, paraphrase) - cos(s, negation), e5 with the `passage: ` prefix
and normalized vectors. margin < 0 means the model puts the norm closer to its
negation than to its paraphrase.

Usage:
    python -m src.diagnostics.d1_margin --lang ru --tag pilot
"""
import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-base"
SEED = 42
ROOT = Path(__file__).resolve().parents[2]


def load_triples(path, lang, include_unchecked=False):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["lang"] == lang]
    if not include_unchecked:
        rows = [r for r in rows if r.get("status") != "NEEDS_KZ_CHECK"]
    return rows


def bootstrap_ci(x, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(m):
    m = np.asarray(m)
    return {
        "n": int(len(m)),
        "mean": float(m.mean()),
        "mean_ci95": bootstrap_ci(m),
        "median": float(np.median(m)),
        "frac_negative": float((m < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=ROOT / "data/diagnostics/d1_triples.jsonl")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--include-unchecked", action="store_true",
                    help="use Kazakh drafts not yet verified by the owner (preliminary numbers only)")
    args = ap.parse_args()

    triples = load_triples(args.triples, args.lang, args.include_unchecked)
    if not triples:
        raise SystemExit(f"no verified triples for lang={args.lang}; see --include-unchecked")
    model = SentenceTransformer(args.model)

    def enc(texts):
        return model.encode(["passage: " + t for t in texts], normalize_embeddings=True,
                            batch_size=32, convert_to_numpy=True)

    S = enc([t["sentence"] for t in triples])
    P = enc([t["paraphrase"] for t in triples])
    N = enc([t["negation"] for t in triples])
    cos_p = (S * P).sum(1)
    cos_n = (S * N).sum(1)
    margin = cos_p - cos_n

    # Robustness control: a one-span, meaning-preserving edit at the same place as the
    # negation edit. Rules out "the paraphrase just has less lexical overlap".
    has_min = all("min_paraphrase" in t for t in triples)
    if has_min:
        M = enc([t["min_paraphrase"] for t in triples])
        cos_m = (S * M).sum(1)
        margin_min = cos_m - cos_n

    per = []
    by_type = defaultdict(list)
    by_mech = defaultdict(list)
    for i, (t, cp, cn, m) in enumerate(zip(triples, cos_p, cos_n, margin)):
        row = {"tid": t["tid"], "neg_type": t["neg_type"], "cos_paraphrase": float(cp),
               "cos_negation": float(cn), "margin": float(m)}
        if has_min:
            row["cos_min_paraphrase"] = float(cos_m[i])
            row["margin_min"] = float(margin_min[i])
        if "neg_mechanism" in t:
            row["neg_mechanism"] = t["neg_mechanism"]
            by_mech[t["neg_mechanism"]].append((m, margin_min[i] if has_min else None))
        per.append(row)
        by_type[t["neg_type"]].append(m)

    out = {
        "diagnostic": "D1_negation_margin",
        "config": {"model": args.model, "prefix": "passage: ", "normalized": True, "lang": args.lang,
                   "triples_file": str(Path(args.triples).relative_to(ROOT)), "seed": SEED,
                   "bootstrap_resamples": 1000, "include_unchecked": args.include_unchecked,
                   "date": date.today().isoformat()},
        "summary": summarize(margin),
        "cos_paraphrase_mean": float(cos_p.mean()),
        "cos_negation_mean": float(cos_n.mean()),
        "by_neg_type": {k: summarize(v) for k, v in sorted(by_type.items())},
        "per_triple": per,
    }
    if by_mech:
        out["by_neg_mechanism"] = {
            k: {"margin": summarize([a for a, _ in v]),
                **({"margin_min": summarize([b for _, b in v])} if has_min else {})}
            for k, v in sorted(by_mech.items())}
    if has_min:
        out["control_min_paraphrase"] = {
            "definition": "cos(s, min_paraphrase) - cos(s, negation); min_paraphrase = one-span synonym edit",
            "cos_min_paraphrase_mean": float(cos_m.mean()),
            "summary": summarize(margin_min),
        }
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    stem = f"d1_margin_{args.tag}_{args.lang}"
    (res / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(min(margin.min(), margin_min.min() if has_min else 0) - 0.005,
                       max(margin.max(), margin_min.max() if has_min else 0) + 0.005, 25)
    ax.hist(margin, bins=bins, color="#4a6fa5", edgecolor="white", alpha=0.85,
            label=f"vs natural paraphrase ({out['summary']['frac_negative']:.0%} < 0)")
    if has_min:
        frac = out["control_min_paraphrase"]["summary"]["frac_negative"]
        ax.hist(margin_min, bins=bins, color="#e1a33b", edgecolor="white", alpha=0.75,
                label=f"vs one-word synonym edit ({frac:.0%} < 0)")
    ax.axvline(0, color="#c0392b", lw=1.5, ls="--")
    ax.set_xlabel("margin = cos(s, paraphrase) − cos(s, negation)   (<0: negation is closer)")
    ax.set_ylabel("triples")
    ax.set_title(f"D1 Negation Margin, e5-base, {args.lang}, n={len(margin)}")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(res / f"{stem}.png", dpi=150)

    keys = ("summary", "cos_paraphrase_mean", "cos_negation_mean", "control_min_paraphrase", "by_neg_mechanism")
    print(json.dumps({k: out[k] for k in keys if k in out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Paired comparison of NLI variants for H5 and H6 (bootstrap over items, 1000 resamples, seed 42).

    H5  does Turkish (b vs a) or machine-translated Kazakh (d vs a) help Kazakh NLI?
        metric: accuracy on the Kardeş kk report half and on the Kazakh manual set
    H6  do entailment counter-examples reduce false contradictions (a vs c)?
        metric: Range-negation Error = share of entailment pairs with a negated range or numeric
        bound that are predicted as contradiction (held-out synthetic, manual set)
    base   the starting model against a and c on the same items, so every rate shares one denominator
Manual sets are also scored without the pairs listed in DOUBTFUL (suffix -clean).

Usage:
    python -m src.rerank.nli_compare
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.rerank.nli_eval import Classifier, eval_sets

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints"
BASE = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
N, SEED = 1000, 42
RANGE_KINDS = {"range", "number"}
COMPARISONS = [("H5", "a", "b"), ("H5", "a", "d"), ("H6", "a", "c"), ("base", "base", "a"), ("base", "base", "c")]
# manual pairs whose annotator label goes against the annotation guideline (checked by hand)
DOUBTFUL = {"v019", "v030", "v035"}


def predictions(variant, sets):
    clf = Classifier(BASE if variant == "base" else str(CKPT / f"nli-{variant}"))
    return {name: clf.probs([r["premise"] for r in rows], [r["hypothesis"] for r in rows]).argmax(1)
            for name, rows in sets.items()}


def metric_vectors(rows, pred):
    """Per-item 0/1 vectors: correct, and (on range entailments only) predicted contradiction."""
    y = np.array([r["label"] for r in rows])
    correct = (pred == y).astype(float)
    rng = np.array([r.get("kind") in RANGE_KINDS and r["label"] == 0 for r in rows])
    return {"accuracy": correct, "range_negation_error": (pred[rng] == 2).astype(float) if rng.any() else None}


def paired(x, y, rng):
    idx = rng.integers(0, len(x), size=(N, len(x)))
    diffs = y[idx].mean(1) - x[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(y.mean() - x.mean()), "n": len(x),
            "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0)}


def main():
    sets = eval_sets()
    sets = {k: v for k, v in sets.items() if k.startswith(("kardes", "synthetic", "manual"))}
    for name in [k for k in sets if k.startswith("manual")]:
        sets[f"{name}-clean"] = [r for r in sets[name] if r["pair_id"] not in DOUBTFUL]
    preds = {v: predictions(v, sets) for v in ["base", *"abcd"]}
    out = {"date": dt.date.today().isoformat(), "bootstrap": {"n": N, "seed": SEED}, "comparisons": []}
    for hyp, x, y in COMPARISONS:
        for name, rows in sets.items():
            mx, my = metric_vectors(rows, preds[x][name]), metric_vectors(rows, preds[y][name])
            for metric in ("accuracy", "range_negation_error"):
                if mx[metric] is None:
                    continue
                res = paired(mx[metric], my[metric], np.random.default_rng(SEED))
                out["comparisons"].append({"hypothesis": hyp, "a": x, "b": y, "set": name, "metric": metric, **res})
                print(f"{hyp} {y} vs {x} {name:18s} {metric:21s} {res['mean_a']:.3f} -> {res['mean_b']:.3f} "
                      f"diff {res['diff']:+.3f} [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]"
                      f"{' *' if res['excludes_0'] else ''} (n={res['n']})")
    path = ROOT / f"results/nli_h5_h6_{out['date']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

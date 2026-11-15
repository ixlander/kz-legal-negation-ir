"""Paired comparison of NLI variants for H5 and H6 (bootstrap over items, 1000 resamples, seed 42).

    H5  does Turkish (b vs a) or machine-translated Kazakh (d vs a) help Kazakh NLI?
        metric: accuracy on the Kardeş kk report half and on the Kazakh manual set
    H6  do entailment counter-examples reduce false contradictions (a vs c)?
        metric: Range-negation Error = share of entailment pairs with a negated range or numeric
        bound that are predicted as contradiction (held-out synthetic, manual set)

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
N, SEED = 1000, 42
RANGE_KINDS = {"range", "number"}
COMPARISONS = [("H5", "a", "b"), ("H5", "a", "d"), ("H6", "a", "c")]


def predictions(variant, sets):
    clf = Classifier(str(CKPT / f"nli-{variant}"))
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
    return {"a": float(x.mean()), "b": float(y.mean()), "diff": float(y.mean() - x.mean()), "n": len(x),
            "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0)}


def main():
    sets = eval_sets()
    sets = {k: v for k, v in sets.items() if k.startswith(("kardes", "synthetic", "manual"))}
    preds = {v: predictions(v, sets) for v in "abcd"}
    out = {"date": dt.date.today().isoformat(), "bootstrap": {"n": N, "seed": SEED}, "comparisons": []}
    for hyp, x, y in COMPARISONS:
        for name, rows in sets.items():
            mx, my = metric_vectors(rows, preds[x][name]), metric_vectors(rows, preds[y][name])
            for metric in ("accuracy", "range_negation_error"):
                if mx[metric] is None:
                    continue
                res = paired(mx[metric], my[metric], np.random.default_rng(SEED))
                out["comparisons"].append({"hypothesis": hyp, "a": x, "b": y, "set": name, "metric": metric, **res})
                print(f"{hyp} {y} vs {x} {name:18s} {metric:21s} {res['a']:.3f} -> {res['b']:.3f} "
                      f"diff {res['diff']:+.3f} [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]"
                      f"{' *' if res['excludes_0'] else ''} (n={res['n']})")
    path = ROOT / f"results/nli_h5_h6_{out['date']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

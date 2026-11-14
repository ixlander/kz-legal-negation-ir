"""Paired bootstrap (CLAUDE.md §6): 1000 resamples, seed 42, 95% percentile CI.

Queries are resampled with replacement within each slice (so every resample keeps the
general/negation sizes), and the same resample is applied to both configurations.
"""
import numpy as np

from src.eval.metrics import summary

N, SEED = 1000, 42


def _get(s, path):
    for p in path:
        s = s[p]
    return s


def paired_diff(rows_a, rows_b, path, n=N, seed=SEED):
    """CI of metric(b) - metric(a); rows are aligned by qid. path e.g. ('negation', 'recall@5')."""
    assert [r["qid"] for r in rows_a] == [r["qid"] for r in rows_b]
    rng = np.random.default_rng(seed)
    by_slice = {}
    for i, r in enumerate(rows_a):
        by_slice.setdefault(r["slice"], []).append(i)
    point = _get(summary(rows_b), path) - _get(summary(rows_a), path)
    diffs = []
    for _ in range(n):
        idx = [j for ix in by_slice.values() for j in rng.choice(ix, size=len(ix), replace=True)]
        a, b = _get(summary([rows_a[j] for j in idx]), path), _get(summary([rows_b[j] for j in idx]), path)
        if a is not None and b is not None:
            diffs.append(b - a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "excludes_0": bool(lo > 0 or hi < 0),
            "n_resamples": len(diffs)}

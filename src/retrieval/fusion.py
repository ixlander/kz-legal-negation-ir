"""Reciprocal Rank Fusion: score = Σ w / (k + rank), rank starting at 1.

By default every ranking is fused in full with weight 1. With `depth`, a ranking contributes only
its top `depth` documents, so documents it cannot tell apart (e.g. zero BM25 score) add nothing.
"""
import numpy as np

K = 60


def ranks(scores):
    """Score vector -> 1-based rank of every document (ties broken by index)."""
    r = np.empty(len(scores), dtype=np.int64)
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


def rrf(*score_vectors, k=K, weights=None, depth=None):
    weights = weights or [1.0] * len(score_vectors)
    total = 0.0
    for w, s in zip(weights, score_vectors):
        r = ranks(s)
        part = w / (k + r)
        if depth is not None:
            part[r > depth] = 0.0
        total = total + part
    return total

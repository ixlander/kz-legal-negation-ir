"""Reciprocal Rank Fusion over full rankings: score = Σ 1 / (k + rank), rank starting at 1."""
import numpy as np

K = 60


def ranks(scores):
    """Score vector -> 1-based rank of every document (ties broken by index)."""
    r = np.empty(len(scores), dtype=np.int64)
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


def rrf(*score_vectors, k=K):
    return sum(1.0 / (k + ranks(s)) for s in score_vectors)

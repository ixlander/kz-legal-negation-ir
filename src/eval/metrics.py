"""Per-query metrics (CLAUDE.md §6) from a full score vector over the corpus.

Recall@k is the share of gold chunks in the top k (most questions have one gold, so 0/1).
Twin-above-Gold: the best-ranked twin is above the best-ranked gold; None without twins.
"""
import numpy as np

from src.retrieval.fusion import ranks


def per_query(scores, gold_idx, twin_idx):
    r = ranks(scores)
    g = r[gold_idx]
    best_g = int(g.min())
    out = {"rank_gold": best_g,
           "recall@5": float((g <= 5).mean()),
           "recall@10": float((g <= 10).mean()),
           "mrr@10": 1.0 / best_g if best_g <= 10 else 0.0,
           "twin_above_gold": None, "rank_twin": None}
    if len(twin_idx):
        best_t = int(r[twin_idx].min())
        out["rank_twin"] = best_t
        out["twin_above_gold"] = float(best_t < best_g)
    return out


def summary(rows):
    """rows: per-query dicts with 'slice' -> means per slice + Negation Gap + Twin-above-Gold."""
    out = {}
    for sl in ("general", "negation", "all"):
        sub = [r for r in rows if sl == "all" or r["slice"] == sl]
        if not sub:
            continue
        out[sl] = {m: float(np.mean([r[m] for r in sub])) for m in ("recall@5", "recall@10", "mrr@10")}
        out[sl]["n"] = len(sub)
    if "general" in out and "negation" in out:
        out["negation_gap"] = out["general"]["recall@5"] - out["negation"]["recall@5"]
    tw = [r["twin_above_gold"] for r in rows if r["slice"] == "negation" and r["twin_above_gold"] is not None]
    n_neg = sum(r["slice"] == "negation" for r in rows)
    out["twin_above_gold"] = {"value": float(np.mean(tw)) if tw else None, "n": len(tw),
                              "share_of_negation_with_twin": len(tw) / n_neg if n_neg else None}
    return out

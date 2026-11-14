"""NLI reranker (B3): the chunk is the premise, the question is the hypothesis.

NLI scores (per candidate):
    s1 = P(entailment) - P(contradiction)   polarity-aware; can push down a correct «не вправе» article
    s2 = P(entailment) + P(contradiction)   «decisiveness»: the chunk settles the question either way
NLI alone does not measure topical relevance (and a question is not a proposition), so on its own it
reorders the top-30 badly. It is therefore fused with the retrieval order:
    pure     the NLI score alone (kept as a reference point)
    rrf      reciprocal rank fusion of the B2 rank and the NLI rank, k=60
    lin-λ    min-max normalized B2 score + λ · NLI score
Scoring, fusion and λ are chosen on the dev split.
"""
import numpy as np

from src.rerank.nli_eval import Classifier
from src.retrieval.fusion import K, ranks

SCORINGS = ("s1", "s2")
LAMBDAS = (0.25, 0.5, 1.0)


def load(path):
    return Classifier(str(path))


def nli_scores(clf, query, docs):
    p = clf.probs(list(docs), [query] * len(docs))
    ent, con = p[:, 0], p[:, 2]
    return {"s1": ent - con, "s2": ent + con}


def fusions(base_top, nli, lambdas=LAMBDAS):
    """base_top: B2 scores of the top-k candidates; nli: their NLI score. -> {fusion name: new scores}."""
    span = base_top.max() - base_top.min()
    base_norm = (base_top - base_top.min()) / span if span > 0 else np.zeros_like(base_top)
    out = {"pure": nli, "rrf": 1.0 / (K + ranks(base_top)) + 1.0 / (K + ranks(nli))}
    out.update({f"lin-{lam:g}": base_norm + lam * nli for lam in lambdas})
    return out


def run_names(fusion_names=None):
    names = fusion_names or ["pure", "rrf"] + [f"lin-{lam:g}" for lam in LAMBDAS]
    return [f"{s}-{f}" for s in SCORINGS for f in names]

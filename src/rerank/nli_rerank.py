"""NLI reranker (B3): the chunk is the premise, the question is the hypothesis.

Two scorings, chosen on the dev split:
    S1 = P(entailment) - P(contradiction)   polarity-aware; can push down a correct «не вправе» article
    S2 = P(entailment) + P(contradiction)   «decisiveness»: the chunk settles the question either way
"""
import numpy as np

from src.rerank.nli_eval import Classifier

SCORINGS = ("s1", "s2")


def load(path):
    return Classifier(str(path))


def scores(clf, query, docs, scoring):
    p = clf.probs(list(docs), [query] * len(docs))
    ent, con = p[:, 0], p[:, 2]
    return np.asarray(ent - con if scoring == "s1" else ent + con)

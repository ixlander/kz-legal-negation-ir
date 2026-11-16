"""BM25 (Okapi, k1=1.5, b=0.75) over an inverted index; scores the whole corpus per query.

Analyzers:
    plain  — lowercase word tokens, nothing removed (B1: lexical match as such)
    morph  — Ru: pymorphy3 lemmas, «не»/«ни» kept, fused «не-» split;
             Kz: kazsearch stems + NEG token for negative verb forms, емес/жоқ kept.
             This is B2.
    morph-noneg — morph with every negation token removed: the control that isolates what
             keeping the negation adds.
Tokenized corpora are cached on disk: the morph analyzers take minutes on 30k chunks.
"""
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from src.retrieval import corpus

WORD = re.compile(r"[а-яёәіңғүұқөһa-z0-9]+")
NEGATION_TOKENS = {"ru": {"не", "ни", "нет", "без"}, "kz": {"NEG", "емес", "жоқ"}}


def plain(text):
    return WORD.findall(text.lower())


def analyzer(name, lang):
    if name == "plain":
        return plain
    if name == "morph-noneg":
        morph, drop = analyzer("morph", lang), NEGATION_TOKENS[lang]
        return lambda text: [t for t in morph(text) if t not in drop]
    if lang == "ru":
        from src.retrieval.morph_ru import tokenize
        return tokenize
    from src.retrieval.morph_kz import tokenize_bm25
    return tokenize_bm25


def tokenized_corpus(name, lang, ids, texts, tag=""):
    """name: analyzer (plain|morph); tag: extra cache-file prefix for non-project corpora."""
    if name == "morph-noneg":
        drop = NEGATION_TOKENS[lang]
        return [[t for t in d if t not in drop] for d in tokenized_corpus("morph", lang, ids, texts, tag)]
    path = corpus.INDEX / f"tokens_{tag}{name}_{lang}.jsonl"
    if path.exists():
        cached = [json.loads(l) for l in open(path, encoding="utf-8")]
        if [c["id"] for c in cached] == ids:
            return [c["tok"] for c in cached]
    tok = analyzer(name, lang)
    docs = [tok(t) for t in texts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, d in zip(ids, docs):
            f.write(json.dumps({"id": i, "tok": d}, ensure_ascii=False) + "\n")
    return docs


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        avg = self.len.mean()
        self.norm = k1 * (1 - b + b * self.len / avg)          # per-doc length term
        self.k1 = k1
        self.post = defaultdict(lambda: ([], []))               # term -> (doc idx, tf)
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                self.post[t][0].append(i)
                self.post[t][1].append(tf)
        self.post = {t: (np.array(ix), np.array(tf, dtype=np.float32)) for t, (ix, tf) in self.post.items()}

    def idf(self, t):
        df = len(self.post[t][0])
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens):
        s = np.zeros(self.n, dtype=np.float32)
        for t, qtf in Counter(query_tokens).items():
            if t not in self.post:
                continue
            ix, tf = self.post[t]
            s[ix] += qtf * self.idf(t) * tf * (self.k1 + 1) / (tf + self.norm[ix])
        return s

"""Dense retrieval: normalized embeddings, brute-force cosine (the corpus is small enough).

Document embeddings are cached on disk together with the id list they were computed for;
a changed corpus invalidates the cache.
"""
import json

import numpy as np

from src import models
from src.retrieval import corpus

MAX_LEN = 512        # chunks are ≤ ~300 words; 512 tokens keeps a 6 GB GPU happy


def _model(key):
    m = models.load(key)
    m.max_seq_length = min(m.max_seq_length, MAX_LEN)
    if m.device.type == "cuda":
        m.half()
    return m


def doc_embeddings(key, lang, ids, texts, model=None):
    vec, idf = corpus.INDEX / f"{key}_{lang}.npy", corpus.INDEX / f"{key}_{lang}.ids.json"
    if vec.exists() and idf.exists() and json.loads(idf.read_text(encoding="utf-8")) == ids:
        return np.load(vec), model
    model = model or _model(key)
    # longest first: fewer padded tokens per batch
    order = np.argsort([-len(t) for t in texts])
    emb = models.encode_docs(model, key, [texts[i] for i in order], batch_size=32)
    out = np.empty_like(emb)
    out[order] = emb
    out = out.astype(np.float32)
    vec.parent.mkdir(parents=True, exist_ok=True)
    np.save(vec, out)
    idf.write_text(json.dumps(ids), encoding="utf-8")
    return out, model


def query_embeddings(key, questions, model=None):
    model = model or _model(key)
    return models.encode_queries(model, key, questions).astype(np.float32), model

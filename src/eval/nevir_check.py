"""NevIR sanity check of the rerankers (English, test split).

Each item has two queries that differ by a negation and two documents; pairwise accuracy is the share
of items where q1 ranks doc1 above doc2 AND q2 ranks doc2 above doc1 (random: 25%). The NevIR
reproduction study reports jina-reranker-v2-base-multilingual at 65.2% and a zero-shot NLI cross-encoder
at 30.2%; bi-encoders mostly 6.5–22.6%. Our numbers should land near those for the same model types.

NLI models score a document as premise and the query as hypothesis (S1 and S2, as in B3).

Usage:
    python -m src.eval.nevir_check
"""
import datetime as dt
import gc
import json
from pathlib import Path

import numpy as np
import torch

from src import models
from src.rerank import cross_encoder, nli_rerank

ROOT = Path(__file__).resolve().parents[2]
NEVIR = ROOT / "data/external/nevir/test.jsonl"
NLI_MODELS = {"mdeberta-xnli (start)": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
              "nli-a (tuned)": str(ROOT / "checkpoints/nli-a")}
EMBEDDERS = ["bge-m3", "e5-base"]


def pairwise(s11, s12, s21, s22):
    """s_ij = score of query i for doc j."""
    return float(np.mean((s11 > s12) & (s22 > s21)))


def free():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    items = [json.loads(l) for l in open(NEVIR, encoding="utf-8")]
    q1, q2 = [r["q1"] for r in items], [r["q2"] for r in items]
    d1, d2 = [r["doc1"] for r in items], [r["doc2"] for r in items]
    res = {}

    for key in EMBEDDERS:
        m = models.load(key)
        Q1, Q2 = models.encode_queries(m, key, q1), models.encode_queries(m, key, q2)
        D1, D2 = models.encode_docs(m, key, d1), models.encode_docs(m, key, d2)
        dot = lambda a, b: (a * b).sum(1)
        res[f"bi-encoder {key}"] = pairwise(dot(Q1, D1), dot(Q1, D2), dot(Q2, D1), dot(Q2, D2))
        del m
        free()

    for key in cross_encoder.RERANKERS:
        m = cross_encoder.load(key)
        sc = lambda qs, ds, m=m: np.asarray(m.predict(list(zip(qs, ds)), batch_size=16, show_progress_bar=False))
        res[f"cross-encoder {key}"] = pairwise(sc(q1, d1), sc(q1, d2), sc(q2, d1), sc(q2, d2))
        del m
        free()

    for name, path in NLI_MODELS.items():
        clf = nli_rerank.load(path)
        probs = {(i, j): clf.probs(d, q) for i, q in ((1, q1), (2, q2)) for j, d in ((1, d1), (2, d2))}
        for scoring in nli_rerank.SCORINGS:
            s = {k: (p[:, 0] - p[:, 2]) if scoring == "s1" else (p[:, 0] + p[:, 2]) for k, p in probs.items()}
            res[f"NLI {name} {scoring}"] = pairwise(s[1, 1], s[1, 2], s[2, 1], s[2, 2])
        del clf
        free()

    out = {"dataset": "NevIR test", "n": len(items), "metric": "pairwise accuracy", "random": 0.25,
           "date": dt.date.today().isoformat(), "results": res}
    path = ROOT / f"results/nevir_check_{out['date']}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:40s} {v:.3f}")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

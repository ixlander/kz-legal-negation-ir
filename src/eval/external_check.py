"""External sanity checks (CLAUDE.md §12, 21.10.2026): can our retrievers search at all?

    rulaw   RuLawRetrieval, split tk_hard (716 queries, Russian Labor Code of the RF), chunk
            protocol of the dataset card: an article gets the max score of its chunks,
            nDCG@10 at article level. The card reports on this split: e5-large 0.820,
            e5-small 0.710, BM25 0.480 -> our numbers should land in that range.
    kazqad  KazQAD test (1929 Kazakh questions). The full corpus has 825k passages, too many to
            embed here, so the corpus is subsampled: all positive and negative passages of the
            test queries + 50k random passages (seed 42). Recall@10 / MRR@10 on that subsample
            are NOT comparable to published numbers; they only compare our runs to each other.

Not gold data for the project; used only to see that the models and analyzers work.

Usage:
    python -m src.eval.external_check --dataset rulaw
    python -m src.eval.external_check --dataset kazqad --models bge-m3 e5-base
"""
import argparse
import datetime as dt
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.retrieval import bm25, dense

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data/external"
SEED, KAZQAD_DISTRACTORS = 42, 50_000


def jl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_rulaw():
    chunks = jl(EXT / "ru-law-retrieval/chunks/chunks.jsonl")
    queries = jl(EXT / "ru-law-retrieval/queries/tk_hard.jsonl")
    rel = defaultdict(set)
    for r in jl(EXT / "ru-law-retrieval/qrels/tk_hard.jsonl"):
        if r["score"] > 0:
            rel[r["query-id"]].add(r["corpus-id"])
    return {"lang": "ru", "unit": [c["doc_id"] for c in chunks], "ids": [c["_id"] for c in chunks],
            "texts": [c["text"] for c in chunks], "queries": [(q["_id"], q["text"]) for q in queries
                                                             if q["_id"] in rel], "rel": rel}


def load_kazqad():
    base = EXT / "kazqad-retrieval"
    qs = jl(base / "queries-and-passages/kazqad-queries-and-passages-v1.0-kk-test.jsonl.gz")
    rel, keep = defaultdict(set), {}
    for q in qs:
        for p in q["positive_passages"]:
            rel[q["query_id"]].add(p["docid"])
            keep[p["docid"]] = f"{p.get('title', '')}\n{p['text']}"
        for p in q["negative_passages"]:
            keep[p["docid"]] = f"{p.get('title', '')}\n{p['text']}"
    pool = [r for r in jl(base / "corpus/kazqad-corpus-v1.0-kk.jsonl.gz")
            if r["docid"] not in keep and len(r["text"].strip()) >= 100]   # drop markup-only stubs
    random.Random(SEED).shuffle(pool)
    for r in pool[:KAZQAD_DISTRACTORS]:
        keep[r["docid"]] = f"{r['title']}\n{r['text']}"
    ids = sorted(keep)
    return {"lang": "kz", "unit": ids, "ids": ids, "texts": [keep[i] for i in ids],
            "queries": [(q["query_id"], q["query"]) for q in qs if rel[q["query_id"]]], "rel": rel}


def unit_scores(scores, unit):
    """Chunk scores -> max per unit (article); returns (unit ids, scores)."""
    best = {}
    for u, s in zip(unit, scores):
        if s > best.get(u, -math.inf):
            best[u] = s
    keys = list(best)
    return keys, np.array([best[k] for k in keys])


def metrics(scores, unit, relevant):
    keys, s = unit_scores(scores, unit)
    top = [keys[i] for i in np.argsort(-s)[:10]]
    hits = [k in relevant for k in top]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    first = next((i for i, h in enumerate(hits) if h), None)
    return {"ndcg@10": dcg / idcg, "recall@10": sum(hits) / min(len(relevant), 10),
            "mrr@10": 0.0 if first is None else 1 / (first + 1)}


def run(data, name, model_keys):
    lang, ids, texts, unit = data["lang"], data["ids"], data["texts"], data["unit"]
    qids, qtext = zip(*data["queries"])
    print(f"[{name}] {len(ids)} passages, {len(qids)} queries")
    runs = {}
    for an in ("plain", "morph"):
        index = bm25.BM25(bm25.tokenized_corpus(an, lang, ids, texts, tag=f"ext-{name}-"))
        tok = bm25.analyzer(an, lang)
        runs[f"bm25-{an}"] = [index.scores(tok(q)) for q in qtext]
        print(f"[{name}] bm25 {an} done")
    for key in model_keys:
        D, m = dense.doc_embeddings(f"{key}", f"ext-{name}", ids, texts)
        Q, _ = dense.query_embeddings(key, list(qtext), m)
        runs[key] = list(Q @ D.T)
        print(f"[{name}] {key} done")
    out = {}
    for r, sc in runs.items():
        per = [metrics(s, unit, data["rel"][q]) for q, s in zip(qids, sc)]
        out[r] = {m: float(np.mean([p[m] for p in per])) for m in ("ndcg@10", "recall@10", "mrr@10")}
    return out, len(ids), len(qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["rulaw", "kazqad"], required=True)
    ap.add_argument("--models", nargs="+", default=["bge-m3", "e5-base"])
    a = ap.parse_args()
    data = load_rulaw() if a.dataset == "rulaw" else load_kazqad()
    res, n_docs, n_q = run(data, a.dataset, a.models)
    out = {"dataset": a.dataset, "n_passages": n_docs, "n_queries": n_q, "date": dt.date.today().isoformat(),
           "note": __doc__.split("\n\n")[1].strip(), "results": res}
    if a.dataset == "kazqad":
        out["subsample"] = {"distractors": KAZQAD_DISTRACTORS, "seed": SEED, "min_chars": 100}
    path = ROOT / f"results/external_{a.dataset}_{'+'.join(a.models)}_{out['date']}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for r, m in res.items():
        print(f"{r:12s} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""B0–B2 of the ablation ladder on the full 22-code corpus, per language.

    B0  dense only
    B1  dense + BM25 plain, RRF k=60
    B2  dense + BM25 morph (negation kept), RRF k=60
BM25-only runs are reported for context. Paired bootstrap for B1−B0, B2−B0, B2−B1.

Usage:
    python -m src.eval.run_ablation --queries draft            # agent drafts, smoke test only
    python -m src.eval.run_ablation --queries final            # finalized set, test split
    python -m src.eval.run_ablation --models bge-m3 --langs kz
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.eval.bootstrap import paired_diff
from src.eval.metrics import per_query, summary
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import rrf

ROOT = Path(__file__).resolve().parents[2]
COMPARE = [("B0", "B1"), ("B0", "B2"), ("B1", "B2")]
PATHS = [("general", "recall@5"), ("negation", "recall@5"), ("negation_gap",), ("twin_above_gold", "value")]


def load_queries(which, lang):
    if which == "draft":
        return [json.loads(l) for l in open(ROOT / f"data/eval/queries_draft_{lang}.jsonl", encoding="utf-8")]
    rows = [json.loads(l) for l in open(ROOT / "data/eval/queries.jsonl", encoding="utf-8")]
    return [r for r in rows if r["lang"] == lang and r["split"] == "test"]


def evaluate(scores, queries, pos, ids):
    rows = []
    for q, s in zip(queries, scores):
        m = per_query(s, np.array([pos[c] for c in q["gold_ids"]]), np.array([pos[c] for c in q["twin_ids"]]))
        top = np.argsort(-s)[:5]
        rows.append({"qid": q["qid"], "slice": q["slice"], **m, "top5": [ids[i] for i in top]})
    return rows


def run_lang(lang, which, model_keys):
    ids, texts = corpus.load(lang)
    pos = {c: i for i, c in enumerate(ids)}
    queries = load_queries(which, lang)
    qtext = [q["question"] for q in queries]
    print(f"[{lang}] corpus {len(ids)} chunks, {len(queries)} queries")

    sparse = {}
    for name in ("plain", "morph"):
        index = bm25.BM25(bm25.tokenized_corpus(name, lang, ids, texts))
        tok = bm25.analyzer(name, lang)
        sparse[name] = [index.scores(tok(q)) for q in qtext]
        print(f"[{lang}] bm25 {name} ready")

    out = {"bm25-plain": evaluate(sparse["plain"], queries, pos, ids),
           "bm25-morph": evaluate(sparse["morph"], queries, pos, ids)}
    for key in model_keys:
        D, model = dense.doc_embeddings(key, lang, ids, texts)
        Q, _ = dense.query_embeddings(key, qtext, model)
        dn = list(Q @ D.T)
        out[f"{key}/B0"] = evaluate(dn, queries, pos, ids)
        out[f"{key}/B1"] = evaluate([rrf(d, s) for d, s in zip(dn, sparse["plain"])], queries, pos, ids)
        out[f"{key}/B2"] = evaluate([rrf(d, s) for d, s in zip(dn, sparse["morph"])], queries, pos, ids)
        del model
        print(f"[{lang}] {key} done")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", choices=["draft", "final"], default="draft")
    ap.add_argument("--models", nargs="+", default=["bge-m3", "e5-base"])
    ap.add_argument("--langs", nargs="+", default=["ru", "kz"])
    a = ap.parse_args()

    res = {"config": {"ladder": "B0-B2", "queries": a.queries, "models": a.models, "rrf_k": 60,
                      "bm25": {"k1": 1.5, "b": 0.75}, "bootstrap": {"n": 1000, "seed": 42},
                      "corpus": "all 22 codes per language, title + text", "date": dt.date.today().isoformat()},
           "langs": {}}
    if a.queries == "draft":
        res["config"]["warning"] = "agent draft questions, gold/twin not finalized: smoke test, not for the report"
    for lang in a.langs:
        runs = run_lang(lang, a.queries, a.models)
        L = {"summary": {k: summary(v) for k, v in runs.items()}, "compare": {}, "per_query": runs}
        for key in a.models:
            for x, y in COMPARE:
                L["compare"][f"{key}: {y} vs {x}"] = {"/".join(p): paired_diff(runs[f"{key}/{x}"], runs[f"{key}/{y}"], p)
                                                      for p in PATHS}
        res["langs"][lang] = L

    tag = f"{a.queries}_{'+'.join(a.models)}_{'+'.join(a.langs)}_{dt.date.today().isoformat()}"
    path = ROOT / f"results/ablation_b0-b2_{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'run':22s} {'lang':4s} {'gen R@5':>8s} {'neg R@5':>8s} {'gap':>6s} {'TaG':>5s} {'MRR':>5s}")
    for lang, L in res["langs"].items():
        for k, s in L["summary"].items():
            tag_ = s["twin_above_gold"]["value"]
            print(f"{k:22s} {lang:4s} {s['general']['recall@5']:8.2f} {s['negation']['recall@5']:8.2f} "
                  f"{s['negation_gap']:6.2f} {tag_ if tag_ is None else round(tag_, 2)!s:>5s} {s['all']['mrr@10']:5.2f}")
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

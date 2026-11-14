"""Ablation ladder on the full 22-code corpus, per language.

    B0       dense only
    B1       dense + BM25 plain, RRF k=60
    B2       dense + BM25 morph (negation kept), RRF k=60
    B3'      B2 top-30 reordered by an off-the-shelf cross-encoder (one run per reranker)
    B3-rule  B2 top-30 reordered by the polarity rule (score ± alpha)
Below the top-30 every reranked run keeps the B2 order, so full-ranking metrics stay defined.
BM25-only runs are reported for context. Each step is compared with paired bootstrap.

Usage:
    python -m src.eval.run_ablation --queries draft            # agent drafts, smoke test only
    python -m src.eval.run_ablation --queries final            # finalized set, test split
    python -m src.eval.run_ablation --models bge-m3 --langs kz --rerankers
"""
import argparse
import datetime as dt
import gc
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.bootstrap import paired_diff
from src.eval.metrics import per_query, summary
from src.rerank import cross_encoder, polarity_rule
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import rrf

ROOT = Path(__file__).resolve().parents[2]
TOP_K = 30
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


def top_k(scores, k=TOP_K):
    return np.argsort(-scores, kind="stable")[:k]


def reorder_top(base, top, new_scores):
    """Put `top` above every other document, ordered by new_scores; the rest keep the base order."""
    out = base - base.max() - 1.0
    order = np.argsort(-np.asarray(new_scores), kind="stable")
    out[top[order]] = 1000.0 - np.arange(len(top))
    return out


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lang(lang, which, model_keys, reranker_keys, alpha):
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
    out = {"bm25-plain": evaluate(sparse["plain"], queries, pos, ids),
           "bm25-morph": evaluate(sparse["morph"], queries, pos, ids)}

    b2 = {}
    for key in model_keys:
        D, model = dense.doc_embeddings(key, lang, ids, texts)
        Q, _ = dense.query_embeddings(key, qtext, model)
        dn = list(Q @ D.T)
        del model
        free_gpu()
        b2[key] = [rrf(d, s) for d, s in zip(dn, sparse["morph"])]
        out[f"{key}/B0"] = evaluate(dn, queries, pos, ids)
        out[f"{key}/B1"] = evaluate([rrf(d, s) for d, s in zip(dn, sparse["plain"])], queries, pos, ids)
        out[f"{key}/B2"] = evaluate(b2[key], queries, pos, ids)
        rule = []
        for q, s in zip(qtext, b2[key]):
            top = top_k(s)
            rule.append(reorder_top(s, top, polarity_rule.boost(q, [texts[i] for i in top], s[top], lang, alpha)))
        out[f"{key}/B3-rule"] = evaluate(rule, queries, pos, ids)
        print(f"[{lang}] {key}: B0-B2, B3-rule done")

    for rk in reranker_keys:
        model = cross_encoder.load(rk)
        for key in model_keys:
            runs = []
            for q, s in zip(qtext, b2[key]):
                top = top_k(s)
                runs.append(reorder_top(s, top, cross_encoder.scores(model, q, [texts[i] for i in top])))
            out[f"{key}/B3'-{rk}"] = evaluate(runs, queries, pos, ids)
        del model
        free_gpu()
        print(f"[{lang}] reranker {rk} done")
    return out


def comparisons(runs, model_keys, reranker_keys):
    pairs = [("B0", "B1"), ("B0", "B2"), ("B1", "B2"), ("B2", "B3-rule")]
    pairs += [("B2", f"B3'-{rk}") for rk in reranker_keys]
    return {f"{key}: {y} vs {x}": {"/".join(p): paired_diff(runs[f"{key}/{x}"], runs[f"{key}/{y}"], p) for p in PATHS}
            for key in model_keys for x, y in pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", choices=["draft", "final"], default="draft")
    ap.add_argument("--models", nargs="+", default=["bge-m3", "e5-base"])
    ap.add_argument("--langs", nargs="+", default=["ru", "kz"])
    ap.add_argument("--rerankers", nargs="*", default=list(cross_encoder.RERANKERS))
    ap.add_argument("--alpha", type=float, default=0.01, help="B3-rule boost, to be tuned on dev")
    a = ap.parse_args()

    today = dt.date.today().isoformat()
    res = {"config": {"queries": a.queries, "models": a.models, "rerankers": a.rerankers, "rerank_top_k": TOP_K,
                      "rule_alpha": a.alpha, "rrf_k": 60, "bm25": {"k1": 1.5, "b": 0.75},
                      "bootstrap": {"n": 1000, "seed": 42}, "corpus": "all 22 codes per language, title + text",
                      "date": today},
           "langs": {}}
    if a.queries == "draft":
        res["config"]["warning"] = "agent draft questions, gold/twin not finalized: smoke test, not for the report"
    for lang in a.langs:
        runs = run_lang(lang, a.queries, a.models, a.rerankers, a.alpha)
        res["langs"][lang] = {"summary": {k: summary(v) for k, v in runs.items()},
                              "compare": comparisons(runs, a.models, a.rerankers), "per_query": runs}

    path = ROOT / f"results/ablation_{a.queries}_{'+'.join(a.models)}_{'+'.join(a.langs)}_{today}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'run':34s} {'lang':4s} {'gen R@5':>8s} {'neg R@5':>8s} {'gap':>6s} {'TaG':>5s} {'MRR':>5s}")
    for lang, L in res["langs"].items():
        for k, s in L["summary"].items():
            tag = s["twin_above_gold"]["value"]
            print(f"{k:34s} {lang:4s} {s['general']['recall@5']:8.2f} {s['negation']['recall@5']:8.2f} "
                  f"{s['negation_gap']:6.2f} {'-' if tag is None else f'{tag:.2f}':>5s} {s['all']['mrr@10']:5.2f}")
    print(f"\n-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

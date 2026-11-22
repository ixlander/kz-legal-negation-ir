"""Two follow-up retrieval checks on the test split.

    B3'-over-B0   the off-the-shelf cross-encoders rerank the top 30 of dense search alone (B0), not of
                  the hybrid; tests "a cross-encoder over the dense top 30" as stated
    abbreviations Russian questions with НДС / ИПН / КПН written out in full, as the code text spells them;
                  B0, BM25, B2 and B3' (over B2) with the expanded questions
Every new run is compared with its counterpart by paired bootstrap.

Usage:
    python -m src.eval.extra_runs
"""
import datetime as dt
import json
import re

from src.eval.bootstrap import paired_diff
from src.eval.metrics import summary
from src.eval.run_ablation import ROOT, evaluate, free_gpu, load_queries, reorder_top, top_k
from src.rerank import cross_encoder
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import rrf

MODELS = ["bge-m3", "e5-base"]
ABBREVIATIONS = {"НДС": "налог на добавленную стоимость", "ИПН": "индивидуальный подоходный налог",
                 "КПН": "корпоративный подоходный налог"}
PATHS = [("general", "recall@5"), ("negation", "recall@5"), ("twin_above_gold", "value")]


def expand(question):
    for short, full in ABBREVIATIONS.items():
        question = re.sub(rf"\b{short}\b", full, question)
    return question


def run_lang(lang):
    ids, texts = corpus.load(lang)
    pos = {c: i for i, c in enumerate(ids)}
    queries = load_queries("final", lang, "test")
    forms = {"": [q["question"] for q in queries]}
    if lang == "ru":
        forms["-exp"] = [expand(q["question"]) for q in queries]
    index = bm25.BM25(bm25.tokenized_corpus("morph", lang, ids, texts))
    tok = bm25.analyzer("morph", lang)
    out, cand = {}, {}
    for sfx, qtext in forms.items():
        sparse = [index.scores(tok(q)) for q in qtext]
        out[f"bm25-morph{sfx}"] = evaluate(sparse, queries, pos, ids)
        for key in MODELS:
            D, model = dense.doc_embeddings(key, lang, ids, texts)
            Q, _ = dense.query_embeddings(key, qtext, model)
            del model
            free_gpu()
            b0 = list(Q @ D.T)
            b2 = [rrf(d, s) for d, s in zip(b0, sparse)]
            out[f"{key}/B0{sfx}"] = evaluate(b0, queries, pos, ids)
            out[f"{key}/B2{sfx}"] = evaluate(b2, queries, pos, ids)
            cand[key, sfx] = (qtext, b0, b2)
    for rk in cross_encoder.RERANKERS:
        model = cross_encoder.load(rk)
        for (key, sfx), (qtext, b0, b2) in cand.items():
            bases = {"B0": b0} if sfx == "" else {"B2": b2}
            for base_name, base in bases.items():
                runs = []
                for q, s in zip(qtext, base):
                    top = top_k(s)
                    runs.append(reorder_top(s, top, cross_encoder.scores(model, q, [texts[i] for i in top])))
                out[f"{key}/B3'-{rk}-over-{base_name}{sfx}"] = evaluate(runs, queries, pos, ids)
        del model
        free_gpu()
    return out


def compare(runs, a, b):
    return {"/".join(p): paired_diff(runs[a], runs[b], p) for p in PATHS}


def main():
    final = json.loads((ROOT / "results/final_ablation_test.json").read_text(encoding="utf-8"))
    res = {"config": {"split": "test", "rerank_top_k": 30, "abbreviations": ABBREVIATIONS,
                      "date": dt.date.today().isoformat()}, "langs": {}}
    for lang in ("ru", "kz"):
        runs = run_lang(lang)
        old = final["langs"][lang]["per_query"]
        cmp = {}
        for key in MODELS:
            for name in ("B0", "B2"):
                assert summary(runs[f"{key}/{name}"]) == summary(old[f"{key}/{name}"]), f"{key}/{name} differs from the final run"
            for rk in cross_encoder.RERANKERS:
                new = f"{key}/B3'-{rk}-over-B0"
                runs[f"{key}/B3'-{rk}"] = old[f"{key}/B3'-{rk}"]
                cmp[f"{new} vs {key}/B0"] = compare(runs, f"{key}/B0", new)
                cmp[f"{new} vs {key}/B3'-{rk} (over B2)"] = compare(runs, f"{key}/B3'-{rk}", new)
            if lang == "ru":
                for base in ("B0", "B2"):
                    cmp[f"{key}/{base}-exp vs {key}/{base}"] = compare(runs, f"{key}/{base}", f"{key}/{base}-exp")
                for rk in cross_encoder.RERANKERS:
                    cmp[f"{key}/B3'-{rk}-over-B2-exp vs {key}/B3'-{rk}"] = compare(
                        runs, f"{key}/B3'-{rk}", f"{key}/B3'-{rk}-over-B2-exp")
        res["langs"][lang] = {"summary": {k: summary(v) for k, v in runs.items()}, "compare": cmp, "per_query": runs}
        for k, s in res["langs"][lang]["summary"].items():
            t = s["twin_above_gold"]["value"]
            print(f"{lang} {k:44s} gen {s['general']['recall@5']:.2f} neg {s['negation']['recall@5']:.2f} "
                  f"TaG {'-' if t is None else f'{t:.2f}'}")
    path = ROOT / "results/extra_runs_test.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

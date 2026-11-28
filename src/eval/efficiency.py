"""Model sizes, inference latency and NLI confusion matrices.

    sizes        parameter count of every model in the pipeline (and trainable count of the NLI model)
    latency      per-query time of each stage over the test questions (bge-m3, both languages),
                 after one warm-up query; GPU peak memory with all models loaded
    confusion    3x3 confusion matrices (rows = gold, columns = predicted; entailment, neutral,
                 contradiction) of the start model and the fine-tuned nli-a on the Kardeş kk report half
                 and the hand-written legal pairs, with per-class precision / recall / F1

    python -m src.eval.efficiency
Writes results/efficiency.json and results/figures/nli_confusion.png.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.eval.run_ablation import load_queries
from src.rerank import cross_encoder, nli_eval, nli_rerank
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import rrf

ROOT = Path(__file__).resolve().parents[2]
BASE = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CKPT = ROOT / "checkpoints/nli-a"
LABELS = ["entailment", "neutral", "contradiction"]
TOP_K = 30


def n_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t) * 1000


def latency(encoder, rerankers, nli):
    out = {}
    for lang in ("ru", "kz"):
        ids, texts = corpus.load(lang)
        emb, _ = dense.doc_embeddings("bge-m3", lang, ids, texts, encoder)
        index = bm25.BM25(bm25.tokenized_corpus("morph", lang, ids, texts))
        analyze = bm25.analyzer("morph", lang)
        questions = [q["question"] for q in load_queries("final", lang, "test")]
        stages = {}

        def add(name, ms):
            stages.setdefault(name, []).append(ms)

        for n, question in enumerate([questions[0]] + questions):
            q, t_enc = timed(lambda: dense.query_embeddings("bge-m3", [question], encoder)[0][0])
            b0, t_dense = timed(lambda: emb @ q)
            toks, t_an = timed(lambda: analyze(question))
            sp, t_bm = timed(lambda: index.scores(toks))
            b2, t_rrf = timed(lambda: rrf(b0, sp))
            top = np.argsort(-b2, kind="stable")[:TOP_K]
            cand = [texts[i] for i in top]
            t_rr = {k: timed(lambda m=m: cross_encoder.scores(m, question, cand))[1] for k, m in rerankers.items()}
            t_nli = timed(lambda: nli_rerank.nli_scores(nli, question, cand))[1]
            if n == 0:
                continue                                   # warm-up
            for name, ms in [("dense query encoding", t_enc), ("dense search (30,766 x 1024)", t_dense),
                             ("BM25 analyzer", t_an), ("BM25 scoring", t_bm), ("RRF fusion", t_rrf),
                             ("NLI rerank top-30", t_nli)] + [(f"{k} rerank top-30", v) for k, v in t_rr.items()]:
                add(name, ms)
        out[lang] = {"queries": len(questions), "chunks": len(ids),
                     "stages_ms": {k: {"mean": round(float(np.mean(v)), 1), "median": round(float(np.median(v)), 1)}
                                   for k, v in stages.items()}}
    return out


def confusion(clf, sets):
    res = {}
    for name in ("kardes-kk-report", "manual-ru", "manual-kz"):
        rows = sets.get(name)
        if not rows:
            continue
        probs = clf.probs([r["premise"] for r in rows], [r["hypothesis"] for r in rows])
        y, pred = np.array([r["label"] for r in rows]), probs.argmax(1)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        res[name] = {"n": len(rows), "labels": LABELS, "matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                     "accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f)),
                     "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                               "support": int(s[i])} for i in range(3)}}
    return res


def plot_confusion(conf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(conf)
    sets = [s for s in ("kardes-kk-report", "manual-ru", "manual-kz") if s in conf[models[0]]]
    fig, axes = plt.subplots(len(models), len(sets), figsize=(3.6 * len(sets), 3.4 * len(models)))
    axes = np.atleast_2d(axes)
    short = ["ent", "neu", "con"]
    for i, m in enumerate(models):
        for j, s in enumerate(sets):
            mat = np.array(conf[m][s]["matrix"])
            ax = axes[i, j]
            ax.imshow(mat / mat.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, mat[a, b], ha="center", va="center", fontsize=10)
            ax.set_xticks(range(3), short)
            ax.set_yticks(range(3), short)
            ax.set_title(f"{m} · {s}\nmacro F1 {conf[m][s]['macro_f1']:.2f} (n={conf[m][s]['n']})", fontsize=9)
            if j == 0:
                ax.set_ylabel("gold")
            ax.set_xlabel("predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    torch.manual_seed(42)
    out = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    encoder = dense._model("bge-m3")
    rerankers = {k: cross_encoder.load(k) for k in ("bge-reranker-v2-m3", "jina-v2")}
    nli = nli_rerank.load(CKPT)
    out["parameters"] = {
        "bge-m3 (dense encoder)": n_params(encoder),
        "bge-reranker-v2-m3": n_params(rerankers["bge-reranker-v2-m3"].model),
        "jina-reranker-v2-base-multilingual": n_params(rerankers["jina-v2"].model),
        "mDeBERTa-v3-base NLI (nli-a, all trainable)": n_params(nli.model),
        "of which embedding matrix": n_params(nli.model.get_input_embeddings()),
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out["latency"] = latency(encoder, rerankers, nli)
    if torch.cuda.is_available():
        out["peak_gpu_mem_gb_inference_all_models"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    del rerankers
    sets = nli_eval.eval_sets()
    out["confusion"] = {"nli-a": confusion(nli, sets)}
    del nli
    torch.cuda.empty_cache()
    out["confusion"]["start model"] = confusion(nli_eval.Classifier(BASE), sets)
    (ROOT / "results/efficiency.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    plot_confusion(out["confusion"], ROOT / "results/figures/nli_confusion.png")
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

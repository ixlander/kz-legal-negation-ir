"""Interactive demo of the retrieval pipeline on the full Russian or Kazakh corpus (22 codes).

A question goes through every stage the evaluation uses:
    preprocessing   bge-m3 (XLM-R) subword tokens and ids; BM25 analyzer tokens (lemmas / stems + NEG)
    B0              dense retrieval with bge-m3, cosine over 30,766 chunks
    B2              B0 fused with morphological BM25 by reciprocal rank fusion (k=60)
    B3              the top 30 of B2 reranked by a cross-encoder (bge-reranker-v2-m3) or by the
                    fine-tuned NLI model (s1 = P(entail) - P(contradict), linear fusion with λ=0.25)
For the final top 5 the NLI model's three probabilities are shown as a confidence readout, and every
row carries the polarity the rule assigns to it. A question copied from the evaluation set gets its
gold and twin paragraphs marked.

    python -m app.demo                 # http://127.0.0.1:7860
    python -m app.demo --share         # temporary public link
"""
import argparse
import json
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

from src.rerank import cross_encoder, nli_rerank, polarity_rule
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import rrf

ROOT = Path(__file__).resolve().parents[1]
DENSE, TOP_K, SHOW = "bge-m3", 30, 5
NLI_CKPT = ROOT / "checkpoints/nli-a"
KZ_LETTERS = re.compile(r"[әіңғүұқөһ]", re.I)
RERANKERS = ["bge-reranker-v2-m3 (cross-encoder)", "NLI reranker (nli-a, s1, λ=0.25)", "none"]


class Index:
    def __init__(self, lang, model):
        t = time.perf_counter()
        self.lang = lang
        self.ids, self.texts = corpus.load(lang)
        self.emb, _ = dense.doc_embeddings(DENSE, lang, self.ids, self.texts, model)
        self.bm25 = bm25.BM25(bm25.tokenized_corpus("morph", lang, self.ids, self.texts))
        self.analyze = bm25.analyzer("morph", lang)
        print(f"[{lang}] {len(self.ids)} chunks indexed in {time.perf_counter() - t:.1f}s")


class Pipeline:
    def __init__(self):
        self.encoder = dense._model(DENSE)
        self.index = {lang: Index(lang, self.encoder) for lang in ("ru", "kz")}
        self._cross, self._nli = None, None
        rows = [json.loads(l) for l in open(ROOT / "data/eval/queries.jsonl", encoding="utf-8")]
        self.known = {r["question"].strip(): r for r in rows}
        self.examples = [[r["question"]] for r in rows if r["split"] == "test" and r["slice"] == "negation"][:12]

    def cross(self):
        if self._cross is None:
            self._cross = cross_encoder.load("bge-reranker-v2-m3")
        return self._cross

    def nli(self):
        if self._nli is None:
            self._nli = nli_rerank.load(NLI_CKPT)
        return self._nli

    def run(self, question, lang, reranker):
        question = question.strip()
        if not question:
            raise gr.Error("Enter a question.")
        if lang == "auto":
            lang = "kz" if KZ_LETTERS.search(question) else "ru"
        ix, timing = self.index[lang], {}

        def clock(name, fn):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing[name] = (time.perf_counter() - t) * 1000
            return out

        # preprocessing
        enc = self.encoder.tokenizer(question)
        pieces = self.encoder.tokenizer.convert_ids_to_tokens(enc["input_ids"])
        q_tokens = ix.analyze(question)
        q_neg = polarity_rule.is_negative(question, lang, q_tokens)

        # retrieval
        q = clock("dense encode", lambda: dense.query_embeddings(DENSE, [question], self.encoder)[0][0])
        b0 = clock("dense search", lambda: ix.emb @ q)
        sparse = clock("BM25", lambda: ix.bm25.scores(q_tokens))
        b2 = clock("RRF fusion", lambda: rrf(b0, sparse))
        top = np.argsort(-b2, kind="stable")[:TOP_K]
        cand = [ix.texts[i] for i in top]

        if reranker.startswith("bge"):
            new = clock("rerank top-30", lambda: np.asarray(cross_encoder.scores(self.cross(), question, cand)))
        elif reranker.startswith("NLI"):
            s1 = clock("rerank top-30", lambda: nli_rerank.nli_scores(self.nli(), question, cand)["s1"])
            new = nli_rerank.fusions(b2[top], s1)["lin-0.25"]
        else:
            new = b2[top]
        b3 = top[np.argsort(-new, kind="stable")]
        b3_score = dict(zip(top.tolist(), np.asarray(new, dtype=float).tolist()))

        final = b3[:SHOW]
        probs = clock("NLI readout (top 5)", lambda: self.nli().probs([ix.texts[i] for i in final], [question] * SHOW))

        known = self.known.get(question)
        gold = set(known["gold_ids"]) if known else set()
        twin = set(known["twin_ids"]) if known else set()

        def mark(i):
            cid = ix.ids[i]
            return "GOLD" if cid in gold else "TWIN" if cid in twin else ""

        def table(order, score):
            rows = []
            for r, i in enumerate(order[:SHOW], 1):
                pol = "NEG" if polarity_rule.chunk_polarity(q_tokens, ix.texts[i], lang) else "POS"
                rows.append({"rank": r, "chunk": ix.ids[i], "score": round(float(score(i)), 4), "polarity": pol,
                             "eval label": mark(i), "text": ix.texts[i][:220].replace("\n", " · ")})
            return pd.DataFrame(rows)

        b0_order = np.argsort(-b0, kind="stable")
        t_b0 = table(b0_order, lambda i: b0[i])
        t_b2 = table(top, lambda i: b2[i])
        t_b3 = table(b3, lambda i: b3_score[i])
        nli_df = pd.DataFrame([{"rank": r, "chunk": ix.ids[i], "P(entailment)": round(float(p[0]), 3),
                                "P(neutral)": round(float(p[1]), 3), "P(contradiction)": round(float(p[2]), 3),
                                "eval label": mark(i)} for r, (i, p) in enumerate(zip(final, probs), 1)])

        def where(cids):
            if not cids:
                return "–"
            pos = {cid: r for r, cid in enumerate((ix.ids[i] for i in b3), 1)}
            return ", ".join(f"{c} (B3 rank {pos[c]})" if c in pos else f"{c} (not in top-30)" for c in sorted(cids))

        prep = (f"**Language:** {lang} · **query polarity (rule):** {'NEG' if q_neg else 'POS'}\n\n"
                f"**bge-m3 / XLM-R subwords** ({len(pieces)} tokens incl. `<s>` `</s>`; max length 512, "
                f"no padding for a single query, attention mask all ones):\n\n"
                f"`{' | '.join(pieces)}`\n\n**token ids:** `{enc['input_ids']}`\n\n"
                f"**BM25 analyzer tokens** ({'pymorphy3 lemmas, не kept' if lang == 'ru' else 'kazsearch stems + NEG'}):"
                f" `{' '.join(q_tokens)}`")
        if known:
            prep += (f"\n\n**From the evaluation set** ({known['qid']}, {known['slice']}, {known['split']} split): "
                     f"gold {where(gold)}; twin {where(twin)}")
        times = " · ".join(f"{k}: {v:.0f} ms" for k, v in timing.items())
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        return prep, t_b0, t_b2, t_b3, nli_df, f"**Latency** ({device}, models warm): {times}"


def build(pipe):
    with gr.Blocks(title="Negation-aware legal retrieval") as ui:
        gr.Markdown("## Negation-aware retrieval over the codes of Kazakhstan (Russian / Kazakh)\n"
                    "30,766 paragraphs per language · B0 dense → B2 hybrid → B3 reranked top-30")
        with gr.Row():
            question = gr.Textbox(label="Question", lines=2, scale=4)
            with gr.Column(scale=1):
                lang = gr.Radio(["auto", "ru", "kz"], value="auto", label="Language")
                reranker = gr.Radio(RERANKERS, value=RERANKERS[0], label="B3 reranker")
        go = gr.Button("Search", variant="primary")
        gr.Examples(pipe.examples, inputs=[question], label="Test-split questions (negation slice)")
        prep = gr.Markdown()
        with gr.Tab("B3 reranked (final)"):
            t3 = gr.Dataframe(wrap=True)
        with gr.Tab("B2 hybrid"):
            t2 = gr.Dataframe(wrap=True)
        with gr.Tab("B0 dense"):
            t0 = gr.Dataframe(wrap=True)
        gr.Markdown("**NLI probabilities for the final top 5** (premise = paragraph, hypothesis = question)")
        nli = gr.Dataframe()
        lat = gr.Markdown()
        outs = [prep, t0, t2, t3, nli, lat]
        go.click(pipe.run, [question, lang, reranker], outs)
        question.submit(pipe.run, [question, lang, reranker], outs)
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    pipe = Pipeline()
    pipe.cross()
    pipe.nli()
    # warm-up: the first CUDA call of each model is slow and would distort the latency readout
    for question in ("Может ли работодатель уволить работника?", "Жұмыс беруші қызметкерді жұмыстан шығара ала ма?"):
        for reranker in RERANKERS[:2]:
            pipe.run(question, "auto", reranker)
    build(pipe).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()

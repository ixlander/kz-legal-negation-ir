"""Interactive demo of the retrieval pipeline on the full Russian or Kazakh corpus (22 codes).

A question goes through every stage the evaluation uses:
    preprocessing   bge-m3 (XLM-R) subword tokens and ids; BM25 analyzer tokens (lemmas / stems + NEG)
    B0              dense retrieval with bge-m3, cosine over 30,766 chunks
    B2              B0 fused with morphological BM25 by reciprocal rank fusion (k=60)
    B3              the top 30 of B2 reranked by a cross-encoder (bge-reranker-v2-m3) or by the
                    fine-tuned NLI model (s1 = P(entail) - P(contradict), linear fusion with λ=0.25)
The page shows the final top 5 with the NLI model's three probabilities, how the ranking changed from
stage to stage, the preprocessing, and the latency of each stage. A question copied from the evaluation
set gets its gold and twin paragraphs marked, with their rank at every stage.

    python -m app.demo                 # http://127.0.0.1:7860
    python -m app.demo --share         # temporary public link
"""
import argparse
import html
import json
import re
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from src.rerank import cross_encoder, nli_rerank, polarity_rule
from src.retrieval import bm25, corpus, dense
from src.retrieval.fusion import ranks, rrf

ROOT = Path(__file__).resolve().parents[1]
DENSE, TOP_K, SHOW = "bge-m3", 30, 5
NLI_CKPT = ROOT / "checkpoints/nli-a"
KZ_LETTERS = re.compile(r"[әіңғүұқөһ]", re.I)
RERANKERS = {"Off-the-shelf reranker": "bge", "Our NLI reranker": "nli", "No reranker": "none"}
RERANKER_NAMES = {"bge": "bge-reranker-v2-m3", "nli": "fine-tuned NLI (mDeBERTa)", "none": "none, final = hybrid"}
LANGUAGES = {"Detect": "auto", "Russian": "ru", "Kazakh": "kz"}
CODES = {
    "admin_offences": "Code on Administrative Offences", "admin_procedure": "Administrative Procedural Code",
    "budget": "Budget Code", "civil_general": "Civil Code (General Part)", "civil_special": "Civil Code (Special Part)",
    "civil_procedure": "Civil Procedure Code", "construction": "Construction Code", "criminal": "Criminal Code",
    "criminal_procedure": "Criminal Procedure Code", "customs": "Customs Code", "digital": "Digital Code",
    "ecology": "Environmental Code", "family": "Code on Marriage and Family", "forest": "Forest Code",
    "health": "Health Code", "labor": "Labor Code", "land": "Land Code", "penal": "Penal Enforcement Code",
    "social": "Social Code", "subsoil": "Subsoil Code", "tax": "Tax Code", "water": "Water Code",
}

CSS = """@import url("https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap");
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.demo, .demo * { box-sizing: border-box; }
.demo { --ink: #ece6dc; --soft: #b4ab9e; --faint: #81796e;
  --paper: #2e2a26; --field: #23201d; --line: #3a3530;
  --neutral-bar: #4b453e; --neg-wash: #3a3226; --neg-label: #e3c48d;
  --green-wash: #1e2a21; --green-label: #a6cfae; --green-mark: #74a982;
  --warn-wash: #33231b; --warn-text: #eda57d; --warn-mark: #d27b4b;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace; --serif: "Newsreader", Georgia, serif;
  color: var(--ink); font-size: 16px; line-height: 24px; }
.demo h3 { font-family: var(--serif); font-size: 23px; line-height: 28px; font-weight: 500; margin: 0 0 4px; }
.demo .hint { color: var(--soft); font-size: 15px; line-height: 22px; margin: 0 0 12px; }
.demo .section { margin-top: 24px; }
.demo .summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.demo .panel { background: var(--field); border-radius: 8px; padding: 16px; }
.demo .panel.ok { background: var(--green-wash); }
.demo .panel.warn { background: var(--warn-wash); }
.demo .panel .big { font-family: var(--serif); font-size: 27px !important; line-height: 32px !important; margin: 4px 0 8px; }
.demo .panel.warn .big { color: var(--warn-text); }
.demo .panel.ok .big { color: var(--green-label); }
.demo .path { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-family: var(--mono); font-size: 14px; }
.demo .step { background: var(--paper); border-radius: 6px; padding: 2px 8px; }
.demo .arrow { color: var(--faint); }
.demo .card { background: var(--field); border-radius: 8px; padding: 16px; margin-bottom: 8px; }
.demo .card.gold { background: var(--green-wash); }
.demo .card.twin { background: var(--warn-wash); }
.demo .card-head { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
.demo .rank { font-family: var(--serif); font-size: 24px; line-height: 24px; min-width: 32px; }
.demo .where { font-weight: 500; }
.demo .cid { font-family: var(--mono); font-size: 13px; color: var(--soft); }
.demo .tags { display: flex; gap: 6px; margin-left: auto; flex-wrap: wrap; }
.demo .tag { font-size: 13px; line-height: 18px; padding: 2px 8px; border-radius: 4px; background: var(--paper); color: var(--soft); }
.demo .tag.gold { color: var(--green-label); }
.demo .tag.twin { color: var(--warn-text); }
.demo .text { margin: 8px 0 0 44px; font-family: var(--serif); font-size: 17px; line-height: 26px; max-width: 75ch; }
.demo .text summary { cursor: pointer; list-style: none; }
.demo .text summary::-webkit-details-marker { display: none; }
.demo .text .more { color: var(--warn-text); font-family: system-ui, sans-serif; font-size: 13px; text-decoration: underline; text-underline-offset: 3px; }
.demo .nli { display: flex; gap: 12px; align-items: center; margin: 10px 0 0 44px; font-size: 14px; color: var(--soft); }
.demo .bar { display: flex; width: 160px; height: 8px; border-radius: 4px; overflow: hidden; background: var(--paper); }
.demo .bar .e { background: var(--green-mark); } .demo .bar .n { background: var(--neutral-bar); } .demo .bar .c { background: var(--warn-mark); }
.demo .key { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; }
.demo .grid { font-size: 15px; }
.demo .grow { display: grid; grid-template-columns: 1fr 110px 110px 110px; border-top: 1px solid var(--line); }
.demo .grow > div { padding: 8px; }
.demo .grow.headrow { border-top: none; color: var(--soft); font-weight: 500; }
.demo .grow .num { font-family: var(--mono); text-align: right; color: var(--faint); }
.demo .grow .num.top { color: var(--ink); font-weight: 600; }
.demo .grow.headrow .num { font-family: inherit; color: var(--soft); }
.demo .grow.gold { background: var(--green-wash); } .demo .grow.twin { background: var(--warn-wash); }
.demo .tokens { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
.demo .tok { font-family: var(--mono); font-size: 12px; background: var(--field); border-radius: 4px; padding: 2px 6px; }
.demo .tok.neg { background: var(--neg-wash); color: var(--neg-label); }
.demo .ids { font-family: var(--mono); font-size: 12px; color: var(--soft); word-break: break-all; }
.demo .latency { color: var(--soft); font-size: 14px; }
.demo .empty { color: var(--soft); background: var(--field); border-radius: 8px; padding: 24px; }
.demo .legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 4px; }
.demo .legend div { background: var(--field); border-radius: 8px; padding: 16px 18px; font-size: 15px; line-height: 22px; color: var(--soft); }
.demo .legend b { display: block; color: var(--ink); font-weight: 600; margin-bottom: 2px; }
.gradio-container { background: #181614 !important; }
.gradio-container h1, .gradio-container h2 { font-family: "Newsreader", Georgia, serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
.gradio-container h2 { font-size: 34px !important; line-height: 40px !important; }
.intro p { font-size: 17px !important; line-height: 26px !important; }
.demo .legend b { font-size: 16px; }
::selection { background: #5a4632; color: #ece6dc; }
"""

DARK = {  # one dark palette, used whatever the browser asks for
    "body_background_fill": "#181614", "background_fill_primary": "#181614", "background_fill_secondary": "#23201d",
    "block_background_fill": "#1f1c19", "block_border_color": "#3a3530", "block_label_text_color": "#b4ab9e",
    "block_title_text_color": "#b4ab9e", "block_info_text_color": "#81796e", "border_color_primary": "#3a3530",
    "body_text_color": "#ece6dc", "body_text_color_subdued": "#b4ab9e", "input_background_fill": "#2a2622",
    "input_border_color": "#3a3530", "input_placeholder_color": "#81796e",
    "button_primary_background_fill": "#e7dccb", "button_primary_background_fill_hover": "#f3eadc",
    "button_primary_text_color": "#181614", "button_secondary_background_fill": "#2a2622",
    "button_secondary_background_fill_hover": "#34302b", "button_secondary_text_color": "#ece6dc",
    "button_secondary_border_color": "#3a3530", "checkbox_background_color": "#2a2622",
    "checkbox_background_color_selected": "#e7dccb", "checkbox_border_color": "#5a534b",
    "checkbox_border_color_selected": "#e7dccb", "checkbox_label_background_fill": "#2a2622",
    "checkbox_label_background_fill_selected": "#34302b", "checkbox_label_text_color": "#ece6dc",
    "checkbox_label_text_color_selected": "#ece6dc", "color_accent": "#e7dccb", "color_accent_soft": "#34302b",
    "link_text_color": "#eda57d", "table_border_color": "#3a3530",
}
LEGEND = ('<div class="legend">'
          '<div><b>B0 dense</b>Meaning-based search: bge-m3 turns the question and every paragraph into vectors. '
          'Our baseline.</div>'
          '<div><b>B2 hybrid</b>B0 merged with keyword search (BM25) that keeps «не» and the Kazakh NEG affix. '
          'Our first fix.</div>'
          '<div><b>B3 reranked</b>A second model re-reads the top 30 of B2: the off-the-shelf reranker, or our NLI '
          'model (our second fix).</div></div>')
EMPTY = ('<div class="demo"><div class="empty">Type a question in Russian or Kazakh, or pick one of the test '
         'questions below, then press Search.</div></div>')


def esc(x):
    return html.escape(str(x))


def where(cid):
    """labor:54:2 -> 'Labor Code, art. 54, para. 2'."""
    code, art, *rest = cid.split(":")
    art = art.split("#")[0]
    para = rest[0].split("#")[0] if rest else ""
    return f"{CODES.get(code, code)}, art. {art}" + (f", para. {para}" if para else "")


class Index:
    def __init__(self, lang, model):
        t = time.perf_counter()
        self.ids, self.texts = corpus.load(lang)
        self.pos = {c: i for i, c in enumerate(self.ids)}
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
        neg = [r for r in rows if r["split"] == "test" and r["slice"] == "negation"]
        gen = [r for r in rows if r["split"] == "test" and r["slice"] == "general"]
        self.examples = [[r["question"]] for r in neg[:6] + gen[:2]]

    def cross(self):
        if self._cross is None:
            self._cross = cross_encoder.load("bge-reranker-v2-m3")
        return self._cross

    def nli(self):
        if self._nli is None:
            self._nli = nli_rerank.load(NLI_CKPT)
        return self._nli

    def search(self, question, lang, reranker):
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

        q_tokens = ix.analyze(question)
        q = clock("dense encoding", lambda: dense.query_embeddings(DENSE, [question], self.encoder)[0][0])
        b0 = clock("dense search", lambda: ix.emb @ q)
        sparse = clock("BM25", lambda: ix.bm25.scores(q_tokens))
        b2 = clock("fusion", lambda: rrf(b0, sparse))
        top = np.argsort(-b2, kind="stable")[:TOP_K]
        cand = [ix.texts[i] for i in top]
        if reranker == "bge":
            new = clock("reranking top 30", lambda: np.asarray(cross_encoder.scores(self.cross(), question, cand)))
        elif reranker == "nli":
            s1 = clock("reranking top 30", lambda: nli_rerank.nli_scores(self.nli(), question, cand)["s1"])
            new = nli_rerank.fusions(b2[top], s1)["lin-0.25"]
        else:
            new = b2[top]
        order = np.argsort(-np.asarray(new), kind="stable")
        b3 = top[order]
        rank = {"B0": ranks(b0), "B2": ranks(b2)}
        r3 = rank["B2"].copy()
        r3[b3] = np.arange(1, len(b3) + 1)
        rank["B3"] = r3
        final = b3[:SHOW]
        probs = clock("NLI readout", lambda: self.nli().probs([ix.texts[i] for i in final], [question] * SHOW))
        enc = self.encoder.tokenizer(question)
        return {"lang": lang, "ix": ix, "q_tokens": q_tokens, "rank": rank, "final": final, "probs": probs,
                "score": dict(zip(top.tolist(), np.asarray(new, dtype=float).tolist())), "timing": timing,
                "pieces": self.encoder.tokenizer.convert_ids_to_tokens(enc["input_ids"]),
                "ids": enc["input_ids"], "q_neg": polarity_rule.is_negative(question, lang, q_tokens),
                "known": self.known.get(question.strip())}

    def run(self, question, lang_label, reranker_label):
        question = (question or "").strip()
        if not question:
            raise gr.Error("Type a question first.")
        r = self.search(question, LANGUAGES[lang_label], RERANKERS[reranker_label])
        return render_summary(r, reranker_label), render_results(r), render_ranks(r), render_prep(r)


def labels(r):
    k = r["known"]
    return (set(k["gold_ids"]) if k else set()), (set(k["twin_ids"]) if k else set())


def render_summary(r, reranker_label):
    k, ix, rank = r["known"], r["ix"], r["rank"]
    lang = {"ru": "Russian", "kz": "Kazakh"}[r["lang"]]
    head = (f'<p class="hint">Searched the {lang} corpus, 30,766 paragraphs of 22 codes. '
            f'Reranker: {esc(RERANKER_NAMES[RERANKERS[reranker_label]])}.</p>')
    if not k:
        return (f'<div class="demo">{head}<div class="panel"><h3>New question</h3><p class="hint" style="margin:0">'
                'This question is not in our evaluation set, so there is no annotated answer to compare with. '
                'Read the top 5 below and check them against the question.</p></div></div>')
    gold, twin = labels(r)

    def best(cids, stage):
        return min((int(rank[stage][ix.pos[c]]), c) for c in cids)

    def path(cids):
        return '<span class="arrow">→</span>'.join(
            f'<span class="step">{name} #{best(cids, st)[0]}</span>'
            for st, name in (("B0", "B0 dense"), ("B2", "B2 hybrid"), ("B3", "B3 reranked")))

    g_rank, g_id = best(gold, "B3")
    slice_word = "polarity-sensitive" if k["slice"] == "negation" else "general"
    if twin and best(twin, "B3")[0] < g_rank:
        cls, verdict = "warn", "The opposite rule is ranked above the correct answer"
    elif g_rank <= SHOW:
        cls, verdict = "ok", f"Correct answer found at #{g_rank}"
    else:
        cls, verdict = "warn", f"Correct answer is outside the top 5 (#{g_rank})"
    gold_panel = (f'<div class="panel {cls}"><h3>Correct answer</h3><p class="big">{esc(verdict)}</p>'
                  f'<p class="hint">{esc(where(g_id))}</p><div class="path">{path(gold)}</div></div>')
    if twin:
        t_rank, t_id = best(twin, "B3")
        twin_panel = (f'<div class="panel"><h3>Opposite rule (twin)</h3><p class="big">Ranked #{t_rank}</p>'
                      f'<p class="hint">{esc(where(t_id))}</p><div class="path">{path(twin)}</div></div>')
    else:
        twin_panel = ('<div class="panel"><h3>Opposite rule (twin)</h3><p class="big">None annotated</p>'
                      '<p class="hint">The code has no paragraph on this topic with the opposite rule.</p></div>')
    note = (f'<p class="hint" style="margin-top:12px">Test question {esc(k["qid"])}, {slice_word}. Ranks are '
            'positions in the full list of 30,766 paragraphs; #1 is the top.</p>')
    return f'<div class="demo">{head}<div class="summary">{gold_panel}{twin_panel}</div>{note}</div>'


def render_results(r):
    ix, gold, twin = r["ix"], *labels(r)
    cards = []
    for n, (i, p) in enumerate(zip(r["final"], r["probs"]), 1):
        cid = ix.ids[i]
        kind = "gold" if cid in gold else "twin" if cid in twin else ""
        tags = []
        if kind == "gold":
            tags.append('<span class="tag gold">Correct answer</span>')
        if kind == "twin":
            tags.append('<span class="tag twin">Opposite rule</span>')
        text = ix.texts[i]
        title, _, body = text.partition("\n")
        short = body[:260] + ("…" if len(body) > 260 else "")
        more = (f'<details><summary>{esc(short)} <span class="more">Show all</span></summary>'
                f'<p>{esc(body)}</p></details>') if len(body) > 260 else f'<p style="margin:0">{esc(body)}</p>'
        e, nu, c = (float(x) for x in p)
        cards.append(
            f'<div class="card {kind}"><div class="card-head"><span class="rank">{n}</span>'
            f'<span class="where">{esc(where(cid))}</span><span class="cid">{esc(cid)}</span>'
            f'<div class="tags">{"".join(tags)}</div></div>'
            f'<div class="text"><p class="hint" style="margin:0 0 4px">{esc(title)}</p>{more}</div>'
            f'<div class="nli"><span>NLI</span><div class="bar" role="img" aria-label="entailment {e:.2f}, '
            f'neutral {nu:.2f}, contradiction {c:.2f}"><div class="e" style="width:{e * 100:.1f}%"></div>'
            f'<div class="n" style="width:{nu * 100:.1f}%"></div><div class="c" style="width:{c * 100:.1f}%"></div>'
            f'</div><span><i class="key" style="background:var(--green-mark)"></i>entails {e:.2f}</span>'
            f'<span><i class="key" style="background:var(--neutral-bar)"></i>neutral {nu:.2f}</span>'
            f'<span><i class="key" style="background:var(--warn-mark)"></i>contradicts {c:.2f}</span></div></div>')
    head = ('<h3>Top 5 after reranking (B3)</h3><p class="hint">What an assistant would read. The bar shows how the NLI '
            'model relates each paragraph to the question; a question is not a statement, so it often says '
            'neutral even for the right answer.</p>')
    return f'<div class="demo">{head}{"".join(cards)}</div>'


def render_ranks(r):
    ix, rank, gold, twin = r["ix"], r["rank"], *labels(r)
    shown = []
    for st in ("B3", "B2", "B0"):
        for i in np.argsort(rank[st])[:SHOW]:
            if int(i) not in shown:
                shown.append(int(i))
    for c in sorted(gold | twin):
        if ix.pos[c] not in shown:
            shown.append(ix.pos[c])
    shown.sort(key=lambda i: (rank["B3"][i], rank["B2"][i]))
    rows = []
    for i in shown:
        cid = ix.ids[i]
        kind = "gold" if cid in gold else "twin" if cid in twin else ""
        label = {"gold": " (correct answer)", "twin": " (opposite rule)"}.get(kind, "")
        cells = "".join(f'<div class="num{" top" if rank[st][i] <= SHOW else ""}">#{int(rank[st][i])}</div>'
                        for st in ("B0", "B2", "B3"))
        rows.append(f'<div class="grow {kind}"><div>{esc(where(cid))}{label}</div>{cells}</div>')
    return ('<div class="demo"><h3>How the ranking changed</h3><p class="hint">Every paragraph that reached the top 5 '
            'at any stage; ranks inside the top 5 are bold.</p><div class="grid"><div class="grow headrow">'
            '<div>Paragraph</div><div class="num">B0 dense</div><div class="num">B2 hybrid</div><div class="num">B3 reranked</div>'
            f'</div>{"".join(rows)}</div></div>')


def render_prep(r):
    pieces = "".join(f'<span class="tok">{esc(p)}</span>' for p in r["pieces"])
    neg_tokens = bm25.NEGATION_TOKENS[r["lang"]]
    words = "".join(f'<span class="tok{" neg" if t in neg_tokens else ""}">{esc(t)}</span>' for t in r["q_tokens"])
    analyzer = "pymorphy3 lemmas, stop words removed except не, ни, нет, без" if r["lang"] == "ru" else \
        "kazsearch stems; a negative verb becomes stem + NEG"
    times = " · ".join(f"{k} {v:.0f} ms" for k, v in r["timing"].items())
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return (f'<div class="demo"><h3>What the dense model reads</h3><p class="hint">{len(r["pieces"])} subword tokens '
            '(XLM-R SentencePiece, ▁ marks the start of a word), max length 512, attention mask all ones.</p>'
            f'<div class="tokens">{pieces}</div><p class="ids">token ids: {esc(r["ids"])}</p>'
            f'<h3 class="section">What BM25 reads</h3><p class="hint">{analyzer}. Negation tokens are highlighted. '
            f'Polarity rule on the question: {"negation found" if r["q_neg"] else "no negation"}.</p>'
            f'<div class="tokens">{words}</div>'
            f'<p class="latency section">Latency on {esc(device)}: {esc(times)}.</p></div>')


def build(pipe):
    with gr.Blocks(title="Negation-aware legal retrieval") as ui:
        gr.Markdown("## Do retrievers confuse “may” and “may not”?\n"
                    "Ask a legal question in Russian or Kazakh. The search runs over 30,766 paragraphs of 22 codes "
                    "of Kazakhstan in three stages:", elem_classes=["intro"])
        gr.HTML(f'<div class="demo">{LEGEND}</div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", lines=3, placeholder="Может ли работодатель …?")
                go = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                lang = gr.Radio(list(LANGUAGES), value="Detect", label="Language")
                reranker = gr.Radio(list(RERANKERS), value="Off-the-shelf reranker", label="Reranker for B3",
                                    info="Off-the-shelf: bge-reranker-v2-m3. Ours: fine-tuned NLI model. "
                                         "None: the final list is B2.")
        gr.Examples(pipe.examples, inputs=[question], label="Questions from the test set")
        summary = gr.HTML(EMPTY)
        with gr.Tabs():
            with gr.Tab("Top 5"):
                results = gr.HTML()
            with gr.Tab("Ranking by stage (B0, B2, B3)"):
                table = gr.HTML()
            with gr.Tab("Preprocessing and latency"):
                prep = gr.HTML()
        outs = [summary, results, table, prep]
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
        for reranker in ("bge", "nli"):
            pipe.search(question, "auto", reranker)
    theme = gr.themes.Base(primary_hue="stone", neutral_hue="stone", radius_size="sm", text_size="lg",
                           font=[gr.themes.GoogleFont("IBM Plex Sans"), gr.themes.Font("system-ui"),
                                 gr.themes.Font("sans-serif")],
                           font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), gr.themes.Font("monospace")]).set(
        **{k + suffix: v for suffix in ("", "_dark") for k, v in DARK.items()
           if hasattr(gr.themes.Base(), k + suffix)})
    # the page is designed for one dark palette; the URL parameter keeps Gradio on its dark variables
    force_dark = ("() => { const u = new URL(window.location.href);"
                   " if (u.searchParams.get('__theme') !== 'dark') {"
                   " u.searchParams.set('__theme', 'dark'); window.location.replace(u.href); } }")
    build(pipe).launch(server_port=a.port, share=a.share, css=CSS, theme=theme, js=force_dark)


if __name__ == "__main__":
    main()

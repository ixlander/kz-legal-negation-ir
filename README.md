# kz-legal-negation-ir

**Do retrievers confuse "may" and "may not"? A parallel Russian–Kazakh benchmark for negation in legal retrieval.**

Maxim Abakarov, Balym Sekenova (Kazakh-British Technical University) · paper draft: [`docs/paper/paper.pdf`](docs/paper/paper.pdf)

A retrieval-augmented legal assistant answers from the articles its retriever returns. If the retriever returns a
norm with the opposite polarity ("the employer may" instead of "the employer may not"), the answer contradicts
the law. This repository holds a parallel Russian/Kazakh corpus of the codes of Kazakhstan, a small evaluation set
with opposite-polarity "twins", diagnostics of encoders and tokenizers, and an ablation of two inexpensive fixes:
a BM25 that keeps negation as a token (fused with dense retrieval) and an NLI reranker, against off-the-shelf
cross-encoder rerankers and a polarity rule.

## Main findings

The test set has 19–20 questions per slice and language, so the retrieval results are a pilot: no pre-registered
comparison survives an exact sign test with a Holm correction (`results/stats_extra.md`).

* With bge-m3, the opposite-polarity twin ranks above the correct paragraph for 2 of 14 Russian and 1 of 14 Kazakh
  test questions; off-the-shelf rerankers bring this to 0–1. The NLI reranker is never better than them.
* The negation token in BM25 changes almost nothing: it occurs in 43–45% of all chunks and few questions contain
  a negation word. Where the hybrid helps (multilingual-e5-base on Kazakh), the gain comes from lemmatization.
* Encoders put a norm closer to its negation than to a paraphrase, but further from its negation than from a
  one-word synonym edit (D1). XLM-R isolates the Kazakh negative affix in 21% of 171 forms (D2), with no visible
  cost to Kazakh retrieval.
* Fine-tuning the NLI model on rule-made legal pairs raises synthetic accuracy from 0.68 to 0.97 with no
  detectable gain on hand-written pairs.

Details, intervals and limitations: the paper and [`PROGRESS.md`](PROGRESS.md) (full lab notebook, including an
internal review and an independent review in [`docs/review/`](docs/review/)). Pipeline diagram, dataset card, model
card, training configuration and checkpoint loading: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Demo

```
python -m app.demo          # http://127.0.0.1:7860, needs the cached embeddings and checkpoints/nli-a
```

Type a question in Russian or Kazakh. The page shows the subword tokens and ids, the BM25 tokens (with `NEG`), the top 5
of dense search (B0), the hybrid (B2) and the reranked list (B3, bge-reranker-v2-m3 or the NLI reranker) with scores
and the polarity of each paragraph, the NLI probabilities for the final top 5, and the latency of each stage. For a
question from the evaluation set the gold and twin paragraphs are marked.

## Data

| Set | Size | Where |
|---|---|---|
| Corpus: 22 codes of Kazakhstan with the same Ru/Kz revision, split into paragraphs | 30,766 chunks per language | `data/corpus/{code}_{ru,kz}.jsonl` |
| Evaluation questions (Labor and Tax Codes), gold and twin paragraph ids, dev/test split | 49 pairs | `data/eval/queries.jsonl` |
| D1 sentence triples (norm, paraphrase, negation, one-word edit) | 30 per language | `data/diagnostics/d1_triples.jsonl` |
| D2 Kazakh negative verb forms with affix boundaries | 171 | `data/diagnostics/d2_kz_neg_forms.tsv` |
| Rule-made NLI pairs from the 20 distractor codes | 3,708 ru, 2,901 kz | `data/nli/synthetic_{ru,kz}.jsonl` |
| Hand-written NLI validation pairs on Labor/Tax norms | 50 ru + 50 kz | `data/nli/val_manual.jsonl` |
| Annotation answers (one annotator) | 195 cards | `data/review/answers/` |

The corpus is official legislation from [adilet.zan.kz](https://adilet.zan.kz). Questions, triples and NLI
hypotheses were drafted with an LLM assistant and checked by a native speaker of Kazakh; the paper describes the
protocol and its limits. `data/eval/queries_v2_draft.jsonl` holds 69 unannotated drafts of a planned larger set and is
not used in any result.

Third-party datasets (XNLI, SNLI-kk, Kardeş-NLU, RussianSuperGLUE, KazQAD, RuLawRetrieval, NevIR) are not
versioned; `python -m src.data.fetch_external` downloads them to `data/external/` (KazQAD is gated on Hugging Face:
log in and accept its terms first).

## Repository layout

```
src/
  scrape/        adilet.zan.kz scraper, code list, Ru/Kz alignment check
  retrieval/     corpus loader, dense retrieval, BM25 with Ru/Kz analyzers, RRF fusion
  rerank/        cross-encoders, polarity rule, NLI data / training / evaluation / reranking
  diagnostics/   D1 Negation Margin, D2 Affix Isolation, Kazakh form extraction and cross-check
  eval/          metrics, bootstrap, ablation runner, final pipeline, exact tests, error analysis, external checks
  data/          external downloads, applying annotation answers
  models.py      embedding models and their input conventions
data/            corpus, evaluation set, diagnostics data, NLI data, annotation answers
results/         every number reported (see results/README.md)
notebooks/       analysis.ipynb (reads results/ only) and the script that generates it
app/             Gradio demo of the full pipeline
tools/review/    builders of the annotation pages
docs/            paper (LaTeX), defense deck source, annotation brief, reviews, list of papers read
CLAUDE.md        project plan, hypotheses, metric definitions and decision log (in Russian)
PROGRESS.md      lab notebook with every step and number
```

## Setup

Python 3.12, a CUDA GPU with 6 GB is enough (all runs were on an RTX 4050 laptop GPU).

```
python -m venv .venv
.venv\Scripts\pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

Kazakh tools, cloned next to the repository (`../nlp_tools/`):

* [kazsearch-py](https://github.com/iDynbek/kazsearch-py) (LGPL-3.0): `pip install ../nlp_tools/kazsearch-py`; the
  Kazakh BM25 analyzer reads its lexicon from `../nlp_tools/kazsearch-py/data` or from `KAZSEARCH_DATA`.
* Only for the D2 cross-check: [kaznlp](https://github.com/nlacslab/kaznlp) and
  [apertium-kaz](https://github.com/apertium/apertium-kaz) (GPL-3.0) with `pyhfst`.

## Reproducing the results

Corpus (optional: the corpus is versioned):
```
python -m src.scrape.list_codes
python -m src.scrape.adilet --all
python -m src.scrape.check_alignment --code labor
```

External data and NLI training:
```
python -m src.data.fetch_external
python -m src.rerank.nli_data --per-lang 50000 --synthetic 4000
python -m src.rerank.nli_train --variant a        # also b, c, d; 1–2 h each on the laptop GPU
python -m src.rerank.nli_train --variant a --tag curves --curve-every 200   # repeat with learning curves
python -m src.eval.plot_curves --run a-curves      # results/figures/nli_learning_curves.png
python -m src.eval.efficiency                      # model sizes, latency, NLI confusion matrices
```

Everything from the annotation answers to the final table, diagnostics and NLI evaluation in one run:
```
python -m src.eval.finalize --answers data/review/answers
```

Control runs, follow-ups and analysis:
```
python -m src.eval.run_ablation --queries final --split test --rerankers --alphas 0 \
    --bm25-weights 0.25 0.5 1 2 --statements \
    --nli MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 checkpoints/nli-a --out results/controls_test.json
python -m src.eval.extra_runs
python -m src.eval.stats_extra
python -m src.eval.error_analysis
python -m src.eval.external_check --dataset rulaw
python -m src.eval.external_check --dataset kazqad --models bge-m3 e5-base
python -m src.eval.nevir_check
python notebooks/make_analysis.py
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

Paper: `cd docs/paper && latexmk -xelatex paper.tex` (XeLaTeX, for the Kazakh letters).

All random seeds are 42. Embeddings and tokenized corpora are cached in `data/index/` and NLI checkpoints in
`checkpoints/`; neither is versioned.

## License

Code: MIT (`LICENSE`). Data we created: CC BY 4.0; the legal texts come from adilet.zan.kz (`data/LICENSE.md`).

## Licenses of what we use

Models and data keep their own licenses: jina-reranker-v2 is CC BY-NC 4.0, KazQAD and SNLI-kk are CC BY-SA 4.0,
RuLawRetrieval has mixed terms (see its card), kazsearch-py is LGPL-3.0 and apertium-kaz GPL-3.0.

## Citation

```
@misc{abakarov2026negation,
  title  = {Do Retrievers Confuse ``May'' and ``May Not''? A Parallel {R}ussian--{K}azakh Benchmark for Negation in Legal Retrieval},
  author = {Abakarov, Maxim and Sekenova, Balym},
  year   = {2026},
  url    = {https://github.com/ixlander/kz-legal-negation-ir}
}
```

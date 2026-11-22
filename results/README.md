# Results

Every file here was written by a script in `src/`; names carry the configuration and, where useful, the date.
Files from earlier stages are kept so the history in `PROGRESS.md` stays checkable; the final numbers are
in the files marked **final**.

## Retrieval

| File | What | Script |
|---|---|---|
| **`final_table.json`, `final_table.md`** | Test table B0–B3, dev selections, bootstrap tests for H1–H3 | `src/eval/finalize.py` |
| **`final_ablation_test.json`**, `final_ablation_dev.json` | All runs with per-query ranks on the reviewed question set | `src/eval/run_ablation.py` |
| **`stats_extra.json`, `stats_extra.md`** | Exact sign tests with Holm, Wilson intervals, TaG@5, breakdowns | `src/eval/stats_extra.py` |
| **`controls_test.json`**, `controls_dev.json` | BM25 without negation tokens, fusion weights, zero-shot NLI, questions as statements | `src/eval/run_ablation.py` |
| **`extra_runs_test.json`** | Rerankers over the dense top 30; Russian abbreviations written out | `src/eval/extra_runs.py` |
| **`error_analysis.json`, `error_analysis.md`**, `error_cases.json` | 20 test misses with categories | `src/eval/error_analysis.py` |
| `ablation_draft_*_2026-11-15.json` | Runs on the unreviewed drafts (smoke tests, not reported) | `src/eval/run_ablation.py` |

## Diagnostics

| File | What | Script |
|---|---|---|
| **`d1_pool_{ru,kz}.json`**, `d1_pool_table.md` | D1 Negation Margin and control on six encoders | `src/diagnostics/d1_pool.py` |
| **`d1_margin_final_{ru,kz}.json`** (+ `.png`) | D1 on multilingual-e5-base with bootstrap and per-triple values | `src/diagnostics/d1_margin.py` |
| **`d2_pool.json`**, **`d2_affix_isolation_final.json`** | D2 Affix Isolation on 14 tokenizers / on XLM-R | `src/diagnostics/d2_pool.py`, `d2_affix.py` |
| `d2_crosscheck.json` | Negative forms checked against kaznlp and apertium-kaz | `src/diagnostics/d2_crosscheck.py` |
| `d1_margin_pilot_ru.json`, `*_prelim*`, `d2_affix_isolation_checked154.json`, `d1_compare_ru_kz.json` | Earlier pilot and preliminary runs | same scripts |

## NLI

| File | What | Script |
|---|---|---|
| **`nli_eval_final-*_2026-11-16.json`** | Base model and variants a–d on all validation sets | `src/rerank/nli_eval.py` |
| **`nli_h5_h6_2026-11-22.json`** | Paired comparisons for H5, H6 and against the base model | `src/rerank/nli_compare.py` |
| `nli_train_*_mdeberta.json` | Training logs (epochs, selection F1) | `src/rerank/nli_train.py` |
| `nli_eval_*_2026-11-15.json`, `nli_h5_h6_2026-11-1{5,6}.json`, `nli_train_a.json` | Earlier runs (XLM-R from scratch, pre-review comparisons) | same scripts |

## Checks of the setup and corpus

| File | What | Script |
|---|---|---|
| `external_rulaw_*.json`, `external_kazqad_*.json` | RuLawRetrieval and KazQAD sanity checks | `src/eval/external_check.py` |
| `nevir_check_2026-11-15.json` | NevIR pairwise accuracy | `src/eval/nevir_check.py` |
| `alignment_check_*.json` | Ru/Kz alignment of every code | `src/scrape/check_alignment.py` |
| `review_applied.json` | What the annotation changed | `src/data/apply_review.py` |

`figures/` holds the figures of the paper; they are made by `notebooks/analysis.ipynb`. The other `.png` files
are from the diagnostics scripts and `tools/report/make_figures.py`.

# PROGRESS — Negation-Aware Retrieval for Russian and Kazakh Legal Texts

Log of what is done, with numbers. Every number comes from a script run; the file is named next to it.
Model for D1: `intfloat/multilingual-e5-base`; tokenizer for D2: `xlm-roberta-base` (= e5, identical vocabulary).
Seed 42, bootstrap 1000 resamples, 95% CI. Notebook with all tables and figures: `notebooks/diagnostics_d1_d2.ipynb`.

> **Status markers.** *final* = all inputs verified. *prelim* = contains Kazakh drafts written by the agent
> (`NEEDS_KZ_CHECK`), to be recomputed after the owner's check. Russian D1 is final pending the owner's read-through.

---

## 2026-10-21 (night 20→21) — first own numbers: D1, D2, corpus

### Corpus (MUST, Labor Code done)
* adilet.zan.kz serves each act through a public JSON endpoint (`/api/documents/by-ngr/<NGR>?language=rus|kaz`)
  with the full HTML, so no browser scraping is needed. Script: `src/scrape/adilet.py`.
* **Labor Code K1500000414, edition updated 15.09.2026, identical for Ru and Kz.**
  227 / 227 articles aligned by number, 3 repealed articles dropped, 39 long articles split by points →
  **448 aligned chunks per language** (`data/corpus/labor_{ru,kz}.jsonl`, stats in `labor_stats.json`).
  Median chunk 62 words (Ru) / 61 (Kz); median Kz/Ru length ratio 0.97.
* Cleaning: footnotes («Сноска»/«Ескерту») and tables removed; **189 Kazakh words contained a Latin `i`
  instead of Kazakh `і`** (e.g. `белгiленбесе`) — normalized, since it changes both BM25 terms and tokenization.
* Known issues: art. 1 (definitions, 2 751 words) and art. 16 (list without points) are single long chunks — to be
  split by sub-points in week 1; one chunk (`labor:127-3:4`) has an extra Ru paragraph absent in Kz.
* **Tax Code:** the current one is **K2500000214 (No. 214-VIII of 18.07.2025)**; the 2017 code ended 31.12.2025.
  ~~Kz text ~10× smaller~~ — **corrected 21.10:** that came from the `content_size` metadata field; the actual
  HTML is 3.19 M (Kz) vs 3.34 M (Ru) characters. The Kz version marks article headings with `<h3>` instead of
  `<p><b>`, which the parser must handle. Ru: 848 articles, ~279 k words. Update dates differ (07.10 vs 15.09.2026):
  check that the editions match.

### D1 — Negation Margin (`margin = cos(s, paraphrase) − cos(s, negation)`)
30 modal norms of the Labor Code («вправе / обязан / допускается / запрещается / не может превышать»),
**the same 30 norms in Ru and Kz** (verbatim from the current edition). For each: a natural paraphrase, a negation
(15 remove the negation, 11 insert it, 4 antonym/modality flips), and — as a control — a *minimal paraphrase*:
a one-span synonym edit at the same place as the negation edit («вправе» → «имеет право»).

| lang | margin mean [95% CI] | share margin < 0 | control: min-paraphrase margin [95% CI] | share < 0 | status |
|---|---|---|---|---|---|
| ru | **−0.018** [−0.024, −0.013] | **90%** (27/30) | +0.018 [+0.013, +0.022] | 7% | final* |
| kz | **−0.013** [−0.020, −0.006] | **77%** (23/30) | +0.025 [+0.020, +0.030] | 0% | prelim |

Files: `results/d1_margin_pilot_ru.json`, `results/d1_margin_prelim_kz.json`, `results/d1_compare_ru_kz.json`,
figures `results/d1_distance_ladder.png`, `results/d1_margin_hist_ru_kz.png`. Triples: `data/diagnostics/d1_triples.jsonl`.

Mean distance `1 − cos` from the original sentence:

| edit | ru | kz |
|---|---|---|
| one-word synonym edit | 0.005 | 0.005 |
| **negation** | **0.023** | **0.029** |
| natural paraphrase | 0.041 | 0.043 |

**Reading.** D1 is supported in both languages (mean margin < 0, CI excludes 0, > 20% negative).
e5 is *not blind* to negation — it moves the vector 4–6× more than a synonym edit — but **less than an ordinary
rewording**. A user query is a rewording of the article, so the article with the same words and the opposite
polarity can end up closer than the correct one. This is the retrieval failure we will measure with Twin-above-Gold.

* Ru vs Kz on the same norms, paired: Δmargin (kz − ru) = +0.005 [−0.002, +0.011] → **no difference detected**.
  So at the sentence level the Kazakh problem is not worse than the Russian one.
* Kz by negation mechanism (prelim; small n, CIs overlap): affix (`берілмейді`, n=14) −0.016; particle
  `емес` (n=12) −0.009; lexical (`тыйым салынады`, n=4) −0.017.
* Observation, not a claim (small n): removing a negation hurts more than inserting one —
  Ru −0.025 vs −0.009, Kz −0.020 vs −0.003.

Examples (Ru, most negative margin):

| | text | cos to original |
|---|---|---|
| original (art. 10) | Условия … трудовых договоров **не могут** быть изменены в одностороннем порядке, за исключением случаев, предусмотренных настоящим Кодексом. | |
| negation | Условия … трудовых договоров **могут** быть изменены в одностороннем порядке, … | **0.984** |
| paraphrase | Одна сторона не может сама поменять условия трудового или коллективного договора либо соглашения, кроме случаев, указанных в Кодексе. | 0.938 |
| original (art. 175) | **Не допускается** замена работников, участвующих в забастовке … | |
| negation | **Допускается** замена работников, участвующих в забастовке … | **0.945** |
| paraphrase | Работников, которые бастуют в законном порядке, нельзя заменять другими. | 0.903 |

Example (Kz, prelim): art. 83 «… тынығуының ұзақтығы он екі сағаттан кем **болмайды**» → negation
«… кем **болуы мүмкін**» cos 0.987, paraphrase «… кемінде он екі сағат демалуы керек» cos 0.936.

### D2 — Affix Isolation Rate (prelim)
171 distinct Kazakh negative verb forms (581 occurrences) found in the Kz Labor Code by the regex splitter
`src/retrieval/morph_kz.py` (stem + -ма/-ме/-ба/-бе/-па/-пе + ending). Isolation = a token boundary
**both** before and after the affix.

| measure | value |
|---|---|
| **Kz: affix isolated** (distinct forms, n=171) | **21.1%** |
| Kz: affix isolated, weighted by corpus frequency (n=581) | 24.6% |
| Kz: at least one boundary at the affix | 95.9% |
| Ru control: separate «не» is its own token (n=12 phrases) | 100% |
| Ru control: fused prefix «не-» split off (неуплата, n=30) | 50% |

Where the affix ends up (distinct forms): isolated 21% (`▁бол ма уға`) · starts a token glued to the ending 60%
(`▁бол маған`, `▁бері л мейді`) · ends a token glued to the stem 15% (`▁а спа йтын`) · buried inside one token 4%
(`▁болмайды`, `▁алмайды`). Files: `results/d2_affix_isolation_prelim.json`, `results/d2_patterns.png`,
forms `data/diagnostics/d2_kz_neg_forms.tsv`.

**Reading.** D2 is supported (isolation 21% ≪ 70% threshold). The tokenizer rarely gives the negation its own piece:
it is usually fused with the tense ending into units like `маған` / `мейді` — and `маған` is also the ordinary word
"to me". A morphology-aware BM25 that emits an explicit `NEG` token is therefore a meaningful intervention for Kazakh.
Russian is not trivially clean either: a fused «не-» is split off in only half of the words.

Splitter precision: on the corpus, the first version had false positives `немесе` ("or", 592 occurrences!),
`орынбасары` ("deputy"), `қамауда` ("in custody") — fixed with a stop-list and a stricter conditional ending.
Lexicalized negative forms (`қоспағанда` "except", `келіспеушілік` "disagreement") are kept and flagged for the owner.

### Environment
`.venv` (Python 3.12, torch 2.11+cu128, RTX 4050 6 GB), `requirements.txt`. All scripts run with `python -m src....`.

---

## 2026-10-21 (day) — independent checks instead of manual ones (decision log, CLAUDE.md 12)

### D2 forms cross-checked by two independent Kazakh analyzers
`src/diagnostics/d2_crosscheck.py` → `results/d2_crosscheck.json`. kaznlp (segmentation, negation tag `ETB` with
the exact affix span) and apertium-kaz nightly (`kaz.automorf.hfst` via pyhfst, tag `<neg>`); tools live outside
the repo in `../nlp_tools`.
* kaznlp knows 94 of 171 forms; where it does, **the affix position matches the regex splitter in 88 of 88**
  negative cases. apertium knows 162 of 171.
* **154 / 171 forms (488 / 581 occurrences) confirmed by the tools → status `ok_auto`.** 17 go to the owner:
  7 lexicalized (`қоспағанда`, `келіспеушілік*`), 9 unknown to both tools, `алмаса` (ambiguous), `сұғылмауына`.
* Rule stated up front: an apertium analysis whose lemma already contains stem+affix (`реттелмеген<adj>`) counts as
  negative — the affix is in the word, which is what D2 measures.
* **D2 on the 154 confirmed forms: isolation 20.8%** (freq-weighted 17.8%; the drop vs 24.6% is `қоспағанда`,
  52 occurrences, now pending). `results/d2_affix_isolation_checked154.json`. Conclusion unchanged.

### Ru/Kz alignment checked automatically on all 448 pairs
`src/scrape/check_alignment.py` → `results/alignment_check_labor.json`.
Point/sub-point numbering identical in 99.3% of pairs, all numbers identical in 99.3%, true partner in the
cross-lingual top-3 (e5) both ways in 96%. 20 pairs flagged; 15 of them only by rank (numbering and numbers
identical — a near-duplicate neighbour article), **5 structural ones go to the owner**:
* `labor:178` — **the official texts disagree:** Ru refers to «подпункт 22) пункта 1 статьи **52**», Kz to
  «**51**-бабы». Art. 52 is the one with the grounds for termination by the employer, so the Kz text looks wrong.
* `labor:26:1/26:2` — sub-point «2-1)» sits in point 1 in Ru but in point 2 in Kz; `labor:187-1` — Kz has «1.» that
  Ru lacks; `labor:127-3:4` — an extra Ru paragraph.
* Parser fix found by the check: a chapter heading («Глава 18.») inside a `<p>` leaked into `labor:180`.

### Kazakh D1 drafts self-reviewed
9 of 30 Kz triples carry an `agent_doubt` note in the review sheet (4 are modality-based negations by design).
One real error fixed: kz028 paraphrase used «зорлауға» (ambiguous, primarily "to rape") → «күштеуге». Kz D1 after
the fix: −0.0135 [−0.020, −0.006] (was −0.0134).

### For the owner to check (≈30 min total)
1. `data/diagnostics/d2_owner_check.tsv` — **17 forms**: `decision` = `ok` / `not_neg` / `wrong_split`, then
   `python -m src.diagnostics.kz_review apply-d2` and `python -m src.diagnostics.d2_affix`.
2. `data/diagnostics/review_d1_kz.tsv` — 30 Kz triples, start with the 9 that have `agent_doubt`: `status` =
   `ok` / `fixed` (after editing cells) / `drop`, then `python -m src.diagnostics.kz_review apply`.
3. `data/corpus/labor_alignment_flagged.tsv` — **5 pairs**: `aligned_ok` = yes/no (they are the same place in the
   code; differences come from the official texts).
4. Russian triples in `data/diagnostics/d1_triples.jsonl` — quick read of paraphrases/negations.

### Evaluation set — started (drafts only)
`data/eval/queries_draft_ru.{jsonl,tsv}` (generator `data/eval/make_draft_ru.py`): **22 Ru draft questions on the
Labor Code** — 11 negation-sensitive (6 with a twin candidate in a different chunk), 11 general — with candidate
`gold_ids` / `twin_ids` found by corpus search. Owner finalizes gold/twin by reading the code (CLAUDE.md 5.2).
Finding for the metric design: allowed and forbidden often sit **in the same chunk** (art. 76 night work,
77 overtime, 95 recall from leave, 112 downtime), so Twin-above-Gold is undefined there; ~half of the negation
questions may end up without a twin unless such articles are split by sentence. To decide with the owner.

### Next (week 1, per plan)
* Tax Code Ru+Kz with the same scraper; split long no-point articles.
* D1 on 100 triples (extend from 30 per language).
* Draft 50 Ru evaluation questions with gold (owner finalizes gold/twin).

---

## 2026-10-21 (evening) — research, wider pool, full corpus, analyzers

Decisions logged in CLAUDE.md §12 (21.10.2026). Research notes: `docs/research/2026-10-21_tools_models_data.md`.

### Corpus: 22 codes in force, point-level chunks
`python -m src.scrape.adilet --all`. Chunk = point when Ru/Kz point numbering matches, long chunks split by
sub-point groups (≤300 words). **~30.8k aligned chunks per language** (labor 774, tax 3,320); inventory
`data/corpus/codes_inventory.md` (Entrepreneurial Code left out: Ru/Kz editions differ). Alignment check on all
30,766 pairs: 0.86% structural mismatches before number normalization; labor 6 + tax 12 for the owner
(real discrepancies of the official texts, e.g. tax:370 cites art. 399 in Ru vs 339 in Kz); 227 suspect
distractor chunks in `data/corpus/alignment_suspect_distractors.json`.
D1 triples and the Ru eval drafts were remapped to point-level ids; drafts now 24 (13 negation, **10 with a twin**).

### D1 on six embedders (`results/d1_pool_*.json`, table `results/d1_pool_table.md`; kz = prelim)
| model | margin ru | <0 ru | neg. position ru | margin kz* | <0 kz* | neg. position kz* |
|---|---|---|---|---|---|---|
| e5-base | −0.018 | 90% | 0.56 | −0.014 | 77% | 0.57 |
| e5-large-instruct | −0.004 (CI ∋ 0) | 57% | 0.96 | −0.019 | 90% | 0.45 |
| bge-m3 | −0.008 (CI ∋ 0) | 57% | 0.81 | −0.013 (CI ∋ 0) | 60% | 0.87 |
| LaBSE | −0.073 | 97% | 0.30 | −0.074 | 93% | 0.33 |
| Qwen3-Embedding-0.6B | −0.122 | 100% | 0.12 | −0.171 | 100% | 0.09 |
| embeddinggemma-2 | −0.065 | 100% | 0.07 | −0.087 | 100% | −0.02 |
*Negation position* = (d_neg − d_syn) / (d_para − d_syn), scale-free across models: 0 = the negation moves the
vector no more than a one-word synonym edit, 1 = as much as a full rewording.
* **No model has a positive margin**: D1 is not refuted for any of them. For bge-m3 and e5-large-instruct (ru) the
  margin is ≈ 0 (no difference detected) — negation and rewording are equally far, still a retrieval risk.
* Newer general-purpose embedders (Qwen3, embeddinggemma) are the most negation-blind; LaBSE (bitext-trained) too.
* e5-large-instruct improves on Russian (57% negative) but not on Kazakh (90%) — the first sign of a Ru/Kz gap.

### D2 on 14 tokenizers (`results/d2_pool.json`, 154 verified forms)
Isolation: mBERT 47%, Gemma 45%, ruBERT 34%, Qwen3 25%, **XLM-R 21%**, mDeBERTa 10%, LaBSE 7%, Kazakh
monolingual tokenizers 0–2% (frequent negative forms are single tokens: «болмаған», «берілмейді»).
Caveat: the "morph-BPE" Kazakh tokenizer likely expects morphologically pre-segmented input, which its HF
tokenizer does not do — its 0% is not evidence against morphological tokenization.

### BM25 analyzers (B2)
`src/retrieval/morph_ru.py`: pymorphy3 lemmas, fused «не-» split when the rest is a known word of the same POS
group (exception list checked on the Labor Code), negation whitelist instead of a stop list.
`src/retrieval/morph_kz.tokenize_bm25`: kazsearch stems + `NEG`; a negative form gets the stem of its positive form,
so «жол берілмейді» / «жол беріледі» → `жол бері NEG` / `жол бері`.

### External data (`python -m src.data.fetch_external`, not versioned)
KazQAD retrieval, RuLawRetrieval, NevIR, RCB/LiDiRus/TERRa, snli-kazakh, XNLI ru/tr, Kardeş-NLU.

### For the owner (updated)
1. `data/diagnostics/d2_owner_check.tsv` — 17 forms. 2. `data/diagnostics/review_d1_kz.tsv` — 30 Kz triples
(9 with `agent_doubt`). 3. `data/corpus/labor_alignment_flagged.tsv` (6) and `tax_alignment_flagged.tsv` (12).
4. `data/eval/queries_draft_ru.tsv` — 24 Ru draft questions, gold/twin candidates.

## 2026-11-14/15 — full evaluation set drafts, retrieval ladder B0→B3′, NLI data

All retrieval numbers below are on the **agent's draft questions** (gold/twin not yet finalized by a human):
they show that the pipeline works, they are **not** results for the report. Bootstrap: paired, 1000 resamples,
seed 42, 95% CI; "*" = CI excludes 0.

### Evaluation set: 50 question pairs (Ru + Kz), awaiting review
`data/eval/queries_draft_{ru,kz}.jsonl`: 25 Labor + 25 Tax, 25 negation + 25 general, 20 negation questions with
a twin. Kz versions `NEEDS_KZ_CHECK`. All gold/twin ids verified to exist in both corpora. Review page updated
(same URL), brief for the annotator in `docs/annotation/TZ_razmetka.md`.
Retrieval already caught one wrong draft: for «sale of a flat / car owned > 2 y / 1 y» the answering point is
tax:382:5 («доход не образуется…»), not 383:1 / 384:1 — those became the twins.

### Corpus fix
88 chunk ids were duplicated in 5 distractor codes (point numbering restarts inside an article); repeated
numbers now get a `~2` suffix. Labor and Tax unchanged.

### Retrieval ladder on drafts (`results/ablation_draft_bge-m3+e5-base_ru+kz_2026-11-15.json`)
Corpus: all 22 codes per language (30 766 chunks). Recall@5 general / negation, Twin-above-Gold (20 twins):

| run | Ru gen | Ru neg | Ru TaG | Kz gen | Kz neg | Kz TaG |
|---|---|---|---|---|---|---|
| BM25 morph only | 0.70 | 0.52 | 0.30 | 0.84 | 0.72 | 0.10 |
| bge-m3 B0 | 0.96 | 0.72 | 0.20 | 0.96 | 0.94 | 0.05 |
| bge-m3 B2 | 0.88 | 0.76 | 0.20 | 0.92 | 0.84 | 0.10 |
| bge-m3 B3′ bge-reranker | 1.00 | 0.84 | 0.10 | 0.96 | 0.90 | 0.05 |
| bge-m3 B3-rule (α=0.01) | 0.72 | 0.36 | 0.50 | 0.72 | 0.28 | 0.40 |
| e5-base B0 | 0.94 | 0.68 | 0.25 | 0.74 | 0.48 | 0.40 |
| e5-base B2 | 0.86 | 0.64 | 0.20 | 0.92 | 0.64 | 0.15 |
| e5-base B3′ bge-reranker | 1.00 | 0.88 | 0.05 | 0.96 | 0.88 | 0.10 |

* bge-m3 is far stronger than e5-base on Kazakh (negation R@5 0.94 vs 0.48 at B0).
* For bge-m3, B2 vs B0: no difference detected. For e5-base on Kz, B2 lowers Twin-above-Gold 0.40 → 0.15.
* Cross-encoders (B3′) help most on the weaker retriever; e5-base + bge-reranker Ru neg +0.24*, Kz neg +0.24*.
* B3-rule hurts badly (Ru neg −0.40*, Kz −0.56* for bge-m3): most negation questions are phrased positively
  («Можно ли…?») while the answer says «не допускается», so a polarity *match* rule pushes the gold down.
  α is to be tuned on dev (α → 0 would switch it off); this is the behaviour the "deliberately dumb" control
  was meant to expose.

### External sanity checks (`results/external_*.json`)
RuLawRetrieval tk_hard (716 q., chunk protocol, nDCG@10): bge-m3 **0.839**, e5-base **0.762**, BM25 morph
**0.501**, BM25 plain 0.386. Dataset card on the same split: FRIDA 0.854, e5-large 0.820, e5-small 0.710,
BM25 0.480 → our pipeline lands where expected.
KazQAD test (1929 q., subsampled corpus of 52 660 passages, not comparable to published numbers): bge-m3 0.606,
e5-base 0.493, BM25 morph 0.465, BM25 plain 0.348 → the Kazakh analyzer (kazsearch + NEG) helps on general
Kazakh text too, not only on our corpus.

### Rerankers
`jinaai/jina-reranker-v2-base-multilingual` needs `einops` and a helper that transformers 5 removed; a
3-line shim restores it. `BAAI/bge-reranker-v2-m3` loads as is.

### NLI data (`data/nli/`, third-party subsets not versioned)
XNLI ru / tr and SNLI-kk: 50k stratified each; Kardeş-NLU kk dev 1000 / test 2000 (validation only).
Synthetic from the 20 distractor codes (never Labor/Tax): Ru 3 933, Kz 3 107 pairs; contradiction by polarity
flip and modality swap, entailment with a kept negation («не менее → не меньше», «запрещается → не допускается»).
Rules that change word order were dropped (they broke Kazakh grammar); Kz entailment is therefore thin (307
pairs, one pattern). Kz synthetic is `NEEDS_KZ_CHECK`.
Manual validation drafts: `data/nli/val_manual.jsonl`, 50 Ru + 50 Kz pairs from Labor/Tax norms, 19 E / 12 N /
19 C, 21 with range negation; hypotheses avoid the generator's rewrite rules. Status draft / NEEDS_KZ_CHECK.

### Zero-shot NLI baseline: mDeBERTa-v3-base-xnli (`results/nli_eval_mdeberta-xnli-zeroshot_2026-11-15.json`)
Kardeş kk test acc 0.791 · RCB ru acc 0.550 · LiDiRus negation acc 0.843 · synthetic ru 0.902 / kz 0.851.
On the (unchecked) manual drafts: ru 0.66, kz 0.62, Kz range-negation error 0.38 vs 0.00 on Ru — the synthetic
set is too easy; the hand-written set is the discriminative one.

### Next
1. Owner/annotator review (questions, D1 kz, D2 forms, alignment) → final `queries.jsonl` with dev/test split.
2. NLI fine-tuning, variants a–d (training of `a` started), then B3 with S1/S2, scoring chosen on dev.
3. Second annotation batch: manual NLI validation (Kz text + all labels).

## 2026-11-15 — NLI reranker: XLM-R attempt, switch to mDeBERTa, synthetic redesign

Decision log entries of 15.11.2026 in CLAUDE.md §12 (owner's "yes").

### Attempt 1: XLM-R-base from scratch (variant a, as in the original plan)
`FacebookAI/xlm-roberta-base` + 3-way head, XNLI ru 50k + XNLI tr 50k + synthetic ru/kz (107 036 pairs),
2 epochs, lr 2e-5, batch 32, fp16 (`results/nli_train_a.json`, `results/nli_eval_nli-a_2026-11-15.json`).
* First run was killed by the OS-memory guard while saving the checkpoint (fp32, 1.1 GB copy to RAM);
  fixed by saving fp16 weights (547 MB) and freeing memory before saving; batches grouped by length
  (epoch 28 → 20 min).
* Kardeş kk dev macro F1: 0.656 (epoch 1) → 0.684 (epoch 2).

| set | XLM-R variant a | zero-shot mDeBERTa-xnli |
|---|---|---|
| Kardeş kk test (acc) | 0.71 | **0.79** |
| RCB ru | 0.36 | **0.55** |
| LiDiRus negation | 0.60 | **0.84** |
| manual drafts ru / kz | 0.40 / **0.26** | 0.66 / 0.62 |

* On the hand-written pairs the model answers "neutral" almost always (47 of 50 Kz pairs). Cause: a
  shortcut learned from our synthetic data — there every entailment/contradiction differs from its
  premise by one word, and every neutral was a different sentence of the same article, so "few shared
  words → neutral" was a perfect rule. Hand-written hypotheses are paraphrases and break it.
* Evaluation bug: the synthetic pairs used for evaluation were also in training (0.999 was meaningless).
* The XLM-R checkpoint was deleted after the switch; its numbers are kept in the result files above.

### Switch: start from mDeBERTa-v3-base-xnli (already trained on 2.7M multilingual NLI pairs)
Keeps its NLI head; continued fine-tuning on the same data mixes, variants a–d unchanged in meaning;
lr 1e-5, effective batch 32 (16 × 2 accumulation), bf16 (DeBERTa is unstable in fp16; the RTX 4050
supports bf16). Epoch 0 (the start model) is evaluated on Kardeş dev for reference.

### Synthetic data redesign (`src/rerank/nli_data.py`)
* Held-out test split by article (10% of articles); training uses only the train split, evaluation
  only the test split.
* Lexically close neutrals: a tighter numeric bound («не менее трех дней» → «не менее пяти дней»,
  «отыз күннен аспайтын» → «бес күннен аспайтын»); a looser bound is an entailment («не менее трех» →
  «не менее двух»). Numerals are swapped only where the grammar stays intact (Ru genitive forms, Kz
  numerals do not inflect; compound numerals skipped). Distant "same article" neutrals at most ⅓.
* Result: Ru 3 708 pairs (E 1200 / N 908 / C 1600), Kz 2 901 (E 597 / N 704 / C 1600); Kz entailment
  doubled (307 → 597) thanks to the numeric pairs.

### mDeBERTa run 1 failed: NaN from the first step
transformers 5 loads checkpoints in their stored dtype by default; this one is fp16, so AdamW updated fp16
weights and the loss was NaN from batch 1 (the model then predicted one class: F1 0.168). Fix: load in
fp32 explicitly (training and evaluation); training now stops on the first non-finite loss. A 30-step
check after the fix: loss 0.1–1.2, peak GPU 5.75 GB of 6.1.

### Kardeş dev is contaminated for this start model
Model card: mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 was trained on the XNLI validation set; Kardeş
dev is its Kazakh translation (start model: 0.943 on Kardeş dev vs 0.791 on Kardeş test). Epoch selection
now uses the even half of Kardeş test, reporting the odd half; Kardeş dev is not used.

### mDeBERTa run 2 (fp32 weights, bf16 compute): variant a (`results/nli_train_a_mdeberta.json`)
Kardeş select half macro F1: start 0.803 → epoch 1 **0.820** → epoch 2 0.814 (epoch 1 kept). ~50 min/epoch
on the RTX 4050 (fp32 master weights are three times slower than the broken fp16 run).

| set | tuned (a) | start model |
|---|---|---|
| Kardeş kk report half (acc) | **0.807** | 0.779 |
| RCB ru | 0.53 | 0.55 |
| LiDiRus negation | 0.83 | 0.84 |
| synthetic held-out articles ru / kz | **0.97 / 0.99** | 0.68 / 0.74 |
| manual drafts ru / kz (unchecked, n=50) | 0.60 / 0.54 | **0.66 / 0.62** |
| Kz range-negation error on manual drafts | **0.00** | 0.38 |

General NLI kept (RCB, LiDiRus unchanged), Kazakh +2.8 pp. Held-out synthetic is near perfect but shares
templates with training. On the hand-written drafts the tuned model calls paraphrased contradictions
"neutral" more often (Ru C→N 11 of 19); differences are within the ±0.13 noise of 50 unchecked pairs.

### B3 as a pure reranker fails (`results/ablation_draft_b3-nli_bge-m3+e5-base_ru+kz_2026-11-15.json`)
Replacing the B2 order of the top-30 by the NLI score, bge-m3, Recall@5 general / negation:
Ru B2 0.88 / 0.76 → B3 0.64 / 0.56–0.60; Kz B2 0.92 / 0.84 → B3 0.72–0.76 / 0.60 (tuned) and
0.56–0.64 / 0.56–0.58 (start model). General questions drop as much as negation ones, so what is lost is
topical relevance, not polarity: NLI does not score relevance, and a question is not a proposition.

### Change: NLI is fused with the retrieval order (within the plan: B3 scoring is chosen on dev, §4)
`src/rerank/nli_rerank.py`: for S1 and S2, three fusions with the B2 order of the top-30 — `pure`
(reference), `rrf` (B2 rank + NLI rank, k=60), `lin-λ` (min-max normalized B2 score + λ·NLI,
λ ∈ {0.25, 0.5, 1}). The final scoring / fusion / λ is picked on the dev split, never on test.

### B3 with fusion (`results/ablation_draft_b3-fusion_bge-m3+e5-base_ru+kz_2026-11-15.json`, drafts)
Recall@5 general / negation, Twin-above-Gold ("*" = CI of the difference to B2 excludes 0):

| run | bge-m3 Ru | bge-m3 Kz | e5-base Ru | e5-base Kz |
|---|---|---|---|---|
| B2 | 0.88 / 0.76, 0.20 | 0.92 / 0.84, 0.10 | 0.86 / 0.64, 0.20 | 0.92 / 0.64, 0.15 |
| B3 tuned, S1, pure | 0.64 / 0.56, 0.35 | 0.76 / 0.60*, 0.25 | 0.72 / 0.64, 0.30 | 0.82 / 0.56, 0.25 |
| B3 tuned, S2, rrf | 0.90 / 0.80, 0.20 | 0.88 / 0.76, 0.20 | 0.86 / **0.80***, 0.20 | 0.88 / 0.64, 0.20 |
| B3 tuned, S2, lin-0.25 | 0.92 / 0.76, 0.15 | 0.92 / 0.84, 0.10 | 0.86 / 0.72, 0.15 | 0.92 / 0.68, 0.20 |
| B3 start model, S2, rrf | 0.90 / 0.88, 0.20 | 0.72 / 0.80, 0.15 | 0.94 / **0.80***, 0.20 | 0.72 / 0.60, 0.20 |
| B3′ bge-reranker (for reference) | 1.00 / 0.84, 0.10 | 0.96 / 0.90, 0.05 | 1.00 / 0.88, 0.05 | 0.96 / 0.88, 0.10 |

* Fusion removes the collapse: general recall is back at the B2 level (0.86–0.96).
* S2 («decisiveness») is at least as good as S1 on the negation slice and keeps Twin-above-Gold at the B2
  level; S1 raises it (Ru 0.30–0.40, several significant) — the risk noted in the plan: S1 pushes down a
  correct «не допускается» article when the question is phrased positively.
* The only significant negation gains over B2 are on the weaker retriever (e5-base Ru, S2 rrf: 0.64 → 0.80).
  Fine-tuned vs start model: no consistent difference yet.
* No B3 variant reaches the off-the-shelf cross-encoder B3′ on these drafts. Final choice of scoring /
  fusion / λ waits for the dev split; picking the best row here would be selection on test.

### NLI variants a–d on mDeBERTa (`results/nli_train_{a,b,c,d}_mdeberta.json`, `results/nli_eval_nli-*-mdeberta_*.json`)
Trained one after another overnight (b ~55 min, c ~1 h 45, d ~2 h 40). Kardeş select half, best epoch:
a 0.820, b 0.819, c 0.827, d 0.818 (start model 0.803).

| set (accuracy) | start | a | b (no tr) | c (no counter-ex.) | d (+SNLI-kk) |
|---|---|---|---|---|---|
| Kardeş kk report half | 0.779 | 0.807 | 0.812 | 0.797 | 0.801 |
| RCB ru | 0.550 | 0.527 | 0.527 | 0.523 | 0.518 |
| LiDiRus negation | 0.843 | 0.833 | 0.843 | 0.843 | 0.833 |
| synthetic held-out ru / kz | 0.675 / 0.743 | 0.972 / 0.993 | 0.974 / 0.996 | 0.647 / 0.802 | 0.972 / 0.996 |
| manual drafts ru / kz | 0.66 / 0.62 | 0.60 / 0.54 | 0.68 / 0.56 | 0.44 / 0.44 | 0.54 / 0.50 |

Paired bootstrap over items (`src/rerank/nli_compare.py`, `results/nli_h5_h6_2026-11-15.json`):

**H5 (Turkish helps Kazakh) — no difference detected.** b vs a on Kardeş kk report: +0.005
[−0.010, +0.019]; Kazakh manual drafts +0.02 [−0.04, +0.08]. Adding machine-translated SNLI-kk (d vs a):
−0.006 [−0.023, +0.010]. Removing Turkish even helps the Russian manual drafts (+0.08 [+0.02, +0.16]),
on 50 unchecked pairs.

**H6 (entailment counter-examples reduce false contradictions) — supported on held-out synthetic.**
Without them (c vs a), Ru range-negation error 0.009 → 0.609 (+0.60 [+0.50, +0.69]) and Ru held-out
accuracy −0.33; on Kazakh the same pairs are pushed to *neutral* rather than contradiction (Kz numeric
entailments 1.00 → 0.49 correct). The manual drafts agree in direction (Ru −0.16 [−0.28, −0.06]). Note: the
start model already makes almost no range errors on Russian; training on polarity flips *without*
counter-examples is what creates the «не → contradiction» shortcut. Caveats: synthetic test shares
templates with training; manual set still unchecked.

## 2026-11-15 — NevIR sanity check of the rerankers (`results/nevir_check_2026-11-15.json`)
NevIR test (1383 items, English), pairwise accuracy (random 0.25). The NevIR reproduction study reports
jina-reranker-v2 0.652, a zero-shot NLI cross-encoder 0.302, bi-encoders mostly 0.065–0.226.

| model | pairwise acc. |
|---|---|
| bi-encoder bge-m3 / e5-base | 0.267 / 0.188 |
| cross-encoder jina-reranker-v2 | **0.651** (study: 0.652) |
| cross-encoder bge-reranker-v2-m3 | 0.433 |
| NLI mDeBERTa-xnli start, S1 / S2 | 0.570 / 0.341 |
| NLI tuned variant a, S1 / S2 | 0.563 / 0.196 |

* jina matches the published number: the reranker setup (incl. the transformers-5 shim) is correct.
* S1 (entail − contradict) is the polarity-sensitive score: ~0.57, far above the study's English NLI
  baseline. S2 ignores polarity by construction and falls to chance or below — consistent with B3 on our
  drafts, where S2 helped relevance but not polarity. The final B3 choice on dev should weigh this.

## 2026-11-15 — finalization pipeline, tested end to end
After the annotators finish: export the page's `reviews` collection to `data/review/answers/`, then
`python -m src.eval.finalize --answers data/review/answers`. Steps: apply answers
(`src/data/apply_review.py`) → D1 / D2 on checked items → ladder on dev and test (B3-rule α grid
{0, .002, .005, .01, .02}, B3 = NLI variant a in every scoring / fusion) → dev selection → final table on
test (`results/final_table.{json,md}`) with H1–H3 paired tests → NLI evaluation and H5 / H6.
* Dev split: 5 question pairs per slice (negation dev pairs are taken among those with a twin).
* Questions with unknown or empty gold ids are dropped with a warning; gold/twin chunks whose Ru/Kz
  alignment the annotator rejected are reported; fewer than 35 pairs triggers the plan's fallback warning.
* H3 = (Kz gain over B0) − (Ru gain over B0) per question pair, bootstrap over pairs.
* Bug found and fixed on the way: D1 used to keep triples the annotator dropped.
* Tested on 195 fake answers (accept drafts, a few edits, drops, a corrected D2 split, one rejected
  alignment, a bad gold id): every step ran; all test outputs were then reverted, none are kept.

## 2026-11-16 — final run on the reviewed data
Annotation finished (195 cards, `data/review/answers/`). Applied with `src/eval/finalize.py`
(`results/review_applied.json`, no warnings):
* Questions: 49 of 50 accepted (lab_006 dropped: its gold `labor:26:2` is misaligned, see below), no edits;
  final set `data/eval/queries.jsonl` — dev 10 pairs (5 general + 5 negation), test 39 (20 + 19).
* D1 Kz triples: 25 accepted, 5 rewritten (negation and synonym edit); D2: all 17 forms confirmed.
* NLI manual set: all 50 pairs labelled; the annotator kept 94% of the draft labels.
* Alignment: 7 of 18 flagged pairs rejected. Labor Code art. 26: points 1 and 2 are in reverse order in the
  Kazakh edition (Ru 26:1 = Kz 26:2), a real corpus finding; 127-3:4, 191:5, Tax 354:1 split their
  paragraphs differently; art. 178 is the same place with a wrong reference in the official text (52 vs 51)
  and counts as aligned. None of these chunks is a gold or twin of a kept question.

### Retrieval, test split (`results/final_table.md`, `results/final_table.json`)
Dev selection: B3-rule weight α = 0 for both models (every α > 0 lowered dev recall, so the rule is
switched off and equals B2); B3 = S1 + linear fusion λ = 0.25 (bge-m3), S2 + RRF (e5-base).
Recall@5 general / negation (Twin-above-Gold):

| run | bge-m3 Ru | bge-m3 Kz | e5-base Ru | e5-base Kz |
|---|---|---|---|---|
| B0 | 0.95 / 0.79 (0.14) | 0.95 / 0.97 (0.07) | 0.93 / 0.74 (0.29) | 0.78 / 0.47 (0.43) |
| B2 | 0.85 / 0.74 (0.21) | 0.90 / 0.89 (0.07) | 0.82 / 0.68 (0.21) | 0.90 / 0.68 (0.14) |
| B3′ jina | 0.93 / 0.84 (0.07) | 0.90 / 0.92 (0.07) | 0.97 / 0.84 (0.07) | 0.90 / 0.84 (0.14) |
| B3′ bge-reranker | 1.00 / 0.84 (0.07) | 0.95 / 0.92 (0.00) | 1.00 / 0.89 (0.07) | 0.95 / 0.89 (0.07) |
| B3 (NLI, dev-selected) | 0.90 / 0.74 (0.36) | 0.90 / 0.89 (0.07) | 0.88 / 0.79 (0.21) | 0.85 / 0.63 (0.21) |

* **H1 refuted** (difference within CI): B2 vs B0 Negation Gap bge-m3 Ru −0.05 [−0.25, +0.16], Kz +0.03
  [−0.12, +0.18]; e5-base Ru −0.05, Kz −0.09 [−0.37, +0.19]. B2 does raise negation recall of the weak
  model on Kazakh: e5-base Kz 0.47 → 0.68 (+0.21 [+0.05, +0.42]).
* **H2 refuted**: B3 is not better than B3′ anywhere and is worse on e5-base Kz (vs jina −0.21
  [−0.42, −0.05], vs bge-reranker −0.26 [−0.47, −0.11]) and on Twin-above-Gold for bge-m3 Ru (vs
  bge-reranker +0.29 [+0.07, +0.54]). B3-rule is off after dev tuning.
* **H3 partly supported**: with e5-base the B2 gain on negation questions is larger for Kazakh than for
  Russian (+0.26 [+0.11, +0.47], 19 pairs); with B3 +0.11 [0.00, +0.26] (n.s.); with bge-m3 no difference
  (it already reaches 0.97 on Kazakh negation questions at B0).
* bge-m3 is better on Kazakh than on Russian negation questions (0.97 vs 0.79 at B0).

### D1 / D2 on reviewed data
D1 Kazakh, e5-base: margin −0.013 [−0.020, −0.007], 80% of norms closer to their negation (prelim 77%);
no model has a positive margin. D2: XLM-R isolates the affix in 21.1% of 171 forms (all forms now checked).

### NLI on the checked manual set (`results/nli_eval_final-*_2026-11-16.json`, `results/nli_h5_h6_2026-11-16.json`)
Accuracy Ru / Kz (50 pairs each); false contradictions on the 9 Kz range-negation entailments:

| model | manual Ru | manual Kz | Kz range errors |
|---|---|---|---|
| start (mDeBERTa-xnli) | 0.62 | **0.56** | 4 of 9 |
| a (ru + tr + synthetic) | 0.56 | 0.48 | 1 of 9 |
| b (no tr) | **0.64** | 0.50 | 1 of 9 |
| c (no counter-examples) | 0.40 | 0.40 | 0 of 9 |
| d (+ SNLI-kk) | 0.50 | 0.44 | 1 of 9 |

* **H5 not supported**: b vs a on Kazakh manual +0.02 [−0.04, +0.08], on Kardeş +0.005 [−0.010, +0.019].
* **H6**: supported on the held-out synthetic pairs (range error 0.9% vs 60.9%); on the 9 manual Kazakh range
  pairs the counts are too small to test (a 1/9, c 0/9 — without counter-examples the model calls them
  neutral, not contradiction), but c loses 0.16 accuracy on the Russian manual set ([−0.28, −0.06]).
* Fine-tuning cuts Kazakh range errors of the start model (4/9 → 1/9) but lowers overall accuracy on the
  hand-written pairs; the off-the-shelf model is the best on the Kazakh manual set.

## 2026-11-16 — internal review before the paper; change of approach
A review of the whole study (code, data, result files) found no bugs in the metrics, the bootstrap or the
fusion, but several conclusions do not follow from the data. Owner's decision: fix all of it (decision log,
16.11). What changes and why:

**Conclusions restated (the numbers stay, the wording changes)**
* **D1** was reported as "confirmed" from margin = cos(paraphrase) − cos(negation). The paraphrase differs
  from the norm by ~21 tokens (Ru) / ~18 (Kz), the negation by ~1–3, so the negation is closer almost by
  construction. With an edit of the same size (`min_paraphrase`, one-span synonym) the margin is positive:
  e5-base Ru +0.018 [+0.013, +0.022] (7% negative), Kz +0.025 [+0.021, +0.030] (0% negative). So e5 does
  separate a negation from a synonym edit, only less than from a full paraphrase. For the main model bge-m3
  even the original margin is not different from 0 (Ru −0.008 [−0.036, +0.020]). D1 is now reported with the
  size-matched control as the main number.
* **H6** "supported on held-out synthetic pairs" is an artifact: the start model has a Range-negation Error
  of 0.0 on the Russian synthetic test; only variant c has 60.9%, because c keeps the «не менее → менее =
  contradiction» pairs and drops their entailment counterparts — the ablation creates the shortcut it then
  measures. The synthetic test also shares its templates with training. On the hand-written Kazakh range
  pairs: start 4/9, a 1/9, c 0/9. H6 is reported as not testable on 9 pairs.
* **H5** cannot be tested with this base model: by its model card mDeBERTa-2mil7 was already trained on
  machine-translated MNLI/ANLI/FEVER/WANLI/LingNLI in 26 languages including ru and tr (105k pairs each, no
  kk), and variants a and b also differ in size (106k vs 56k pairs).
* **H3** for e5-base: the larger Kazakh gain of B2 is also there on general questions (+0.23 [−0.03, +0.53]),
  so it is a weak dense model rescued by lexical matching, not an effect of negation morphology.
* Fine-tuning on our synthetic pairs lowers accuracy on the hand-written pairs (start 0.62 / 0.56, a 0.56 /
  0.48) while reaching 0.97–0.99 on the synthetic test: the model learned the templates.

**Limits of the question set v1 (49 pairs), to be fixed in v2**
* every negation question is a yes/no question and every general question is a wh-question, so the Negation
  Gap also measures the question type; 22 of 24 negation questions have a prohibiting gold;
* only 2 of 24 negation questions contain a negation word, which makes the polarity rule useless (α = 0)
  and leaves the negation tokens of BM25 nothing to match (see below);
* 19–20 test questions per slice: one question = 0.05 Recall@5, only effects ≥ ~0.25 are detectable; about
  50 intervals without a multiplicity correction; dev of 5 pairs per slice selects noise (different B3
  variants for the two dense models); one gold per question, other relevant chunks are not judged;
* one annotator who saw the model's drafts and labels (hence 94% agreement); Kazakh questions overlap with
  the gold text more than Russian ones (query-token coverage 0.70–0.74 vs 0.54–0.64);
* bge-m3 is at 0.95–0.97 Recall@5 at B0 — no headroom.

**Control runs on v1** (`results/controls_{dev,test}.json`; shared runs reproduce `final_ablation_test.json`)
Recall@5 general / negation (Twin-above-Gold), test:

| run | bge-m3 Ru | bge-m3 Kz | e5-base Ru | e5-base Kz |
|---|---|---|---|---|
| B0 | 0.95 / 0.79 (0.14) | 0.95 / 0.97 (0.07) | 0.93 / 0.74 (0.29) | 0.78 / 0.47 (0.43) |
| B2-noneg (negation tokens removed) | 0.85 / 0.74 (0.21) | 0.90 / 0.89 (0.07) | 0.88 / 0.74 (0.21) | 0.90 / 0.68 (0.14) |
| B2 | 0.85 / 0.74 (0.21) | 0.90 / 0.89 (0.07) | 0.82 / 0.68 (0.21) | 0.90 / 0.68 (0.14) |
| B2, BM25 weight 0.25 | 1.00 / 0.79 (0.14) | 0.95 / 0.97 (0.00) | 0.93 / 0.74 (0.29) | 0.90 / 0.58 (0.43) |
| B2, BM25 weight 0.5 | 1.00 / 0.74 (0.21) | 0.90 / 0.89 (0.07) | 0.93 / 0.74 (0.29) | 0.95 / 0.63 (0.36) |
| B2, BM25 weight 2 | 0.80 / 0.68 (0.29) | 0.90 / 0.84 (0.07) | 0.82 / 0.68 (0.29) | 0.85 / 0.68 (0.07) |

* **Keeping the negation in BM25 changes nothing on v1**: B2 and B2-noneg are identical for bge-m3 and for
  Kazakh, and BM25 alone is identical with and without the negation tokens (Ru 0.58, Kz 0.77 Recall@5). The
  whole B1 → B2 gain is lemmatization / stemming. Expected once we saw that the questions carry no negation
  words; v2 adds questions that do.
* **Equal-weight RRF was the reason B2 fell below B0 for bge-m3**: with BM25 weight 0.25 (dev-selected) the
  hybrid is not below B0 anywhere (Ru general +0.05 [0.00, +0.15], Kz Twin-above-Gold −0.07 [−0.23, 0.00]),
  though not significantly above it either. For e5-base dev picks weight 2 (dev is 10 pairs; the grid is
  reported as a sensitivity analysis, not as a tuned result).
* **B3 with the off-the-shelf NLI model** equals B3 with the fine-tuned one: over the 10 scoring/fusion
  variants mean Recall@5 0.80 vs 0.81 (bge-m3), 0.74 vs 0.76 (e5-base), Twin-above-Gold 0.21 vs 0.20 and
  0.24 vs 0.24; B2 itself: 0.85 / 0.77 and 0.14 / 0.18. Fine-tuning adds nothing to the reranker, and NLI
  reranking is on average below B2.
* **Question → statement** (question particle removed by rule) does not help NLI reranking: mean Recall@5
  0.70 vs 0.80 (start model, bge-m3), 0.73 vs 0.81 (nli-a); single variants vary a lot (Twin-above-Gold
  0.07–0.68), which again shows how noisy 14 twin questions are.

**Next**: NLI seeds 1 and 2 for variants a, b, c (running); question set v2 (150–200 pairs, balanced
design); blind second annotator with Cohen's κ; relevance judgments for the top-5 of every system.

**Question set v2, draft batch 1** (`data/eval/make_draft_v2.py` -> `data/eval/queries_v2_draft.jsonl`): 34 Labor Code
pairs, cells neg-prohibit 17 / neg-permit 11 / neg-explicit 6, 13 with a twin. Agent drafts, Kazakh unchecked;
not used in any metric. Still to draft: Tax Code batch, yes/no questions for the general slice.

**Question set v2, draft batch 2**: + 15 Tax Code pairs; pairs with a permitting gold and no prohibiting norm
nearby form a new cell gen-polar (general slice). Draft now 49 pairs: neg-prohibit 28, neg-permit 9,
neg-explicit 9, gen-polar 3; 16 with a twin. With v1 that is 98 pairs; the permitting and gen-polar cells
are still too small.

**Question set v2, draft batch 3**: + 20 yes/no questions with a permitting gold for the general slice
(11 Labor, 9 Tax). Draft now 69 pairs: neg-prohibit 28, neg-permit 9, neg-explicit 9, gen-polar 23;
with v1 that is 118 pairs.

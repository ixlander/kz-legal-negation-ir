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

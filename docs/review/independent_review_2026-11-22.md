# Independent review: "Do Retrievers Confuse 'May' and 'May Not'? A Parallel Russian–Kazakh Benchmark for Negation in Legal Retrieval"

Reviewer: independent (did not take part in the work). Date: 2026-11-22.
Scope: whole repository at `C:\Users\Admin\Desktop\nlp` (plan, code, data, result files, paper draft, defense deck).
Method: I read the code and data and checked the numbers. Every recomputation below used only stored result files, cached
tokenized corpora (`data/index/tokens_*`) and CPU-side code from `src/`. I did not run model inference or training. Where
I could not verify something without inference, I say so.

---

## 1. Summary of the work

The project asks whether retrievers over Kazakhstan's legislation return the norm with the wrong polarity ("may" vs
"may not"), and whether two cheap components fix this: (i) a morphology-aware BM25 that keeps negation as a token,
fused with a dense retriever by RRF, and (ii) an NLI cross-encoder used as a reranker. Kazakh gets special attention
because it marks negation with a verbal affix.

Artifacts:
* A parallel Ru/Kz corpus of 22 codes (30,766 chunks per language, `data/corpus/*.jsonl`), aligned by shared ids, with an
  automatic alignment check over all pairs (`src/scrape/check_alignment.py`). I recomputed 99.74% matching
  paragraph markers and 99.43% matching numbers over all `results/alignment_check_*.json`, as the paper states.
* 49 parallel question pairs (`data/eval/queries.jsonl`): 25 general + 24 negation per language, dev 10 pairs and test
  39 pairs, 14 of the 19 test negation questions with a twin.
* Diagnostics D1 (Negation Margin, 30 triples per language, 6 encoders) and D2 (Affix Isolation Rate, 171 Kazakh forms,
  14 tokenizers).
* A retrieval ladder B0→B3 with two dense models (bge-m3, multilingual-e5-base), two off-the-shelf cross-encoders, a
  polarity rule, an NLI reranker (mDeBERTa-xnli, fine-tuned in variants a–d), a paired bootstrap, control runs
  (`results/controls_*.json`) and an error analysis on 20 cases.

Main results as reported, all checked against `results/final_table.json`: H1 refuted, H2 refuted, H3 "partly" (e5-base
only, and also on general questions), H5 not supported/untestable, H6 untestable on human pairs, D1 and D2 "held".

### Overall assessment

The engineering is careful. The negative results are reported honestly, and the authors' own internal review of
2026-11-16 (PROGRESS.md, "internal review before the paper") already caught several problems: the D1 edit-size
confound, the H6 artifact, the H5 base-model contamination and the H3 reading. Controls such as B2-noneg, the fusion
weight grid and the zero-shot NLI reranker are the right controls. The jina reranker matches its published NevIR number
(0.651 vs 0.652, `results/nevir_check_2026-11-15.json`), which is good evidence that the reranking setup is correct.
I found no bug in the core metric, RRF or bootstrap code that changes a headline number.

The empirical claims are weaker than the text suggests, though. Of the 14 twin questions per language, the twin appears
in the top 5 while the gold does not in only 0–2 questions in any configuration. Each of the five "found" differences
rests on 4–5 questions that all moved in one direction, and an exact paired sign test gives p = 0.06–0.125 for every
one of them, before any correction for the ~44 tests in the final table. The test questions are verbatim the LLM drafts
on which the B3 fusion design was developed. The annotation protocol gives little real independent verification: 100%
of the LLM questions, golds and twins were accepted unedited, at a median of about 17 s per question card, and all
three NLI label changes the annotator made look wrong. The practitioner takeaway, "put a cross-encoder over the dense
top 30", was never tested as stated, and for bge-m3 it shows no measurable effect.

* **Bachelor thesis defense: accept, 4/5.** The scope, the engineering, the controls and the honesty are above the
  usual level. The weaknesses can be fixed in the text before 1 December by restating claims and adding exact tests,
  with no new annotation.
* **Student workshop (ACL SRW / SIGIR student-track level): borderline, 3/5 (weak accept only after revision).** The
  parallel Ru/Kz statutory corpus and the twin construct are genuine contributions. The retrieval conclusions,
  however, currently rest on a handful of questions with anti-conservative inference, and related work and citations
  of the datasets and models used are missing. In its current form I would score it 2.5.

---

## 2. Major issues

### M1. The benchmark has almost no statistical power for the construct it targets

**What.** The polarity construct is measured on 14 twin questions per language, and very few of them ever show a
polarity error inside the top-k.

**Evidence.** From `results/final_ablation_test.json` (`per_query`, fields `rank_gold` and `rank_twin`), I counted
"polarity-induced misses", meaning twin rank ≤ 5 and gold rank > 5, over the 14 twin questions:

| run | Ru TaG (full) | Ru twin≤5 & gold>5 | Kz TaG (full) | Kz twin≤5 & gold>5 |
|---|---|---|---|---|
| bge-m3 B0 | 2/14 | 1/14 | 1/14 | 0/14 |
| bge-m3 B2 | 3/14 | 2/14 | 1/14 | 1/14 |
| bge-m3 B3′ bge-reranker | 1/14 | 1/14 | 0/14 | 0/14 |
| bge-m3 B3 NLI | 5/14 | 2/14 | 1/14 | 0/14 |
| e5-base B0 | 4/14 | 0/14 | 6/14 | 0/14 |
| e5-base B2 | 3/14 | 1/14 | 2/14 | 0/14 |

* bge-m3 B0 has Recall@5 = 1.00 on all 18 Labor Code test questions in **both** languages. All of its misses are in
  the Tax Code (Ru tax-abbreviated 0.64, tax-other 0.90; Kz 0.86 and 1.00). For the main model, half of the
  benchmark carries no information.
* Wilson 95% intervals for the absolute TaG values in the abstract are 2/14 → [0.04, 0.40] and 1/14 → [0.01, 0.31].

**Why it matters.** "Polarity errors are a smaller problem than we expected for a strong dense retriever" (Discussion)
and "a strong retriever confuses polarity in 1 or 2 of 14 questions" (deck, `takeaway.html`) are claims about rates
that the data cannot pin down. The intervals are compatible with a 30–40% error rate.

**What to do.** Report absolute rates with binomial (Wilson) intervals. Add the top-k polarity-miss count next to TaG.
State in the abstract that the polarity phenomenon shows up in ≤ 2 questions per cell. Present the retrieval results
as a pilot study.

### M2. Statistical inference is anti-conservative, and the five "found" differences do not survive an exact paired test

**What.** `src/eval/bootstrap.py` uses a percentile bootstrap over 19–20 binary outcomes. When all discordant
questions move in one direction, the percentile interval excludes 0 almost automatically. With 4 improving questions
out of 19, a resample contains none of them with probability (15/19)^19 ≈ 0.011 < 0.025.

**Evidence.** These are exact two-sided sign tests on discordant pairs, computed from `per_query` (up = B better,
down = A better):

| difference called "found" (paper, Results) | up / down | exact sign test p |
|---|---|---|
| e5-base Kz, B2 vs B0, negation R@5 (+0.21 [0.05, 0.42]) | 4 / 0 | 0.125 |
| e5-base Kz, B3 vs jina, negation R@5 (−0.21) | 0 / 4 | 0.125 |
| e5-base Kz, B3 vs bge-reranker, negation R@5 (−0.26) | 0 / 5 | 0.0625 |
| bge-m3 Ru, B3 vs bge-reranker, TaG (+0.29 [0.07, 0.54]) | 4 / 0 (twin-above flips) | 0.125 |
| e5-base H3 B2, negation (+0.26 [0.11, 0.47]) | 5 pairs Kz-only gain / 0 opposite | 0.0625 (Wilcoxon 0.025) |

`results/final_table.json` alone contains about 44 interval tests (H1: 3 × 4, H2: 6 × 4, H3: 8), and the
controls add more, all without a multiplicity correction. The paper concedes this only in the Limitations section.

**Why it matters.** The abstract's only positive finding ("BM25 helps … 0.47 to 0.68") and the slide `nliworse.html`
("one of the few differences that is statistically clear") rest on 4 questions.

**What to do.** Report the discordant counts and an exact McNemar/sign test or a paired permutation test for every
comparison, and apply Holm or Benjamini–Hochberg over the pre-registered family (H1–H3). Use BCa or a test-inversion
interval if a bootstrap is kept. Re-word all five findings as "suggestive".

### M3. Test questions were used during method development, and gold labels were revised from retrieval output

**What.** All 98 final queries (question text, `gold_ids`, `twin_ids`) are byte-identical to the agent drafts in
`data/eval/queries_draft_{ru,kz}.jsonl`. I compared them and got 98/98 identical. These drafts were used on 2026-11-15
to develop B3. PROGRESS.md records "B3 as a pure reranker fails" on the drafts, followed by "Change: NLI is fused with
the retrieval order": the RRF and lin-λ fusions, the λ grid and the α grid for B3-rule were all introduced after
seeing those results. The dev/test split was assigned only on 2026-11-16 by `src/data/apply_review.py`. In addition,
the gold of tax_001/002 was changed after a retrieval run (commit `edb3073`: "gold is 382:5 … 383:1 / 384:1 become
twins"; PROGRESS: "Retrieval already caught one wrong draft"). All 2026-11-16 controls were run on the test split after
the final results were known.

**Why it matters.** The paper says the hypotheses were fixed before the evaluation set was finished. That is true for
the hypotheses, but the B3 method family and its tuning grid were shaped on the very questions that now form 80% of
the test set. Selecting gold from retriever output also biases the set toward retrievable gold.

**What to do.** Disclose this in the paper (Systems and Limitations). Mark the B2-wX, B2-noneg, "decl" and zero-shot
B3 runs as exploratory post-hoc analyses. For v2, freeze the method before the questions are seen.

### M4. The annotation provides weak independent verification, and the manual NLI set contains likely label errors

**Evidence.**
* `results/review_applied.json`: 49/50 question cards "ok" and 1 "drop", with **zero edits** to any question, Kazakh
  translation, gold or twin (raw answers in `data/review/answers/reviews/q-*.json`).
* The `updated` timestamps in the answer files give a median gap of **17.1 s** between consecutive question cards
  (50 cards in 54 min), **21.5 s** per NLI pair, **7.6 s** per D1 triple and 3.8 s per D2 form. The annotation brief
  budgets about 1.5 h for the questions (`docs/annotation/TZ_razmetka.md`). Seventeen seconds is not enough to check a
  Ru question, its Kz translation, the gold and the twin against the code. Caveat: timestamps record the last save,
  not the time spent reading.
* All three NLI label changes the annotator made (`data/nli/val_manual.jsonl`, `label_draft` → `label`) seem to go
  the wrong way under the brief's own definitions:
  * v019: premise "paid not less often than once a month", hypothesis "wages may be paid once every two months".
    Changed from contradiction to **entailment**.
  * v030: premise "the employer is not entitled to demand documents not provided by this article", hypothesis "the
    employer may always demand any documents at its discretion". Changed from contradiction to **entailment**.
  * v035: premise "recall from leave is not allowed for workers under 18, pregnant women, …", hypothesis "a worker
    over 18 can be recalled from leave without consent". Changed from neutral to **entailment**.

  The label is shared by the Ru and Kz rows, so 3/50 labels (6%) are probably wrong in both languages. This may have
  been a UI misunderstanding.
* The annotator is a co-author and saw the LLM labels ("Метка Claude показана для сверки", brief §5).

**Why it matters.** The paper's contribution 1 says "checked by a native speaker". The model differences on the
manual set are 3–4 items (base 0.62/0.56 vs a 0.56/0.48), the same size as the suspected label noise.

**What to do.** Before the defense: say exactly what was checked and how (time, label visibility, 100% acceptance),
and flag v019, v030 and v035. Recompute the manual-set accuracies with these three items corrected or excluded; this
needs only prediction inference, seconds on a GPU. For the paper: a blind second annotator and Cohen's κ, as planned
in the decision log of 16.11.

### M5. Twin-above-Gold over the full ranking mixes retrieval failure with polarity confusion

**What.** `src/eval/metrics.py::per_query` compares the best twin and gold ranks over all 30,766 chunks.

**Evidence.** The e5-base Kz B0 TaG of 6/14 comes from (gold, twin) ranks (1096, 32), (70, 48), (2, 1), (2073, 498),
(42, 21) and (74, 32). Only one twin is in the top 5, and in that case the gold is at rank 2. The 0.43 → 0.14 drop
from B0 to B2 is mostly the gold climbing from very deep ranks, not polarity. In the other direction, TaG counts benign
cases: bge-m3 B0 ru_tax_005 has gold at 2 and twin at 1, so both are in the top 5 and a reader model would see both.

**Why it matters.** TaG is presented as "the key metric" (deck `parts.html`), and the e5-base TaG differences feed the
H3 narrative and the ladder slides.

**What to do.** Define TaG@k: twin in the top k and ranked above the gold. Or condition on both being in the top k, or
report the polarity-induced miss shown in M1. Keep full-ranking TaG as a secondary metric.

### M6. The question design has confounds that drive several conclusions

1. **Slice = question type** (acknowledged): all negation questions are yes/no and all general ones are wh-questions.
2. **Gold polarity imbalance.** In 22 of 24 Ru negation questions the gold contains a negation cue (I recomputed this
   with the rule's regex, `src/rerank/polarity_rule.py::RU_NEG`). The exceptions are ru_tax_005 and ru_tax_010. Most
   questions are phrased positively. Consequences:
   * B3-rule as designed (boost on matching polarity) systematically demotes the gold, so dev picks α = 0 and the
     H2 comparison against B3-rule is vacuous: `final_table.json` reports "B3 vs B3-rule-0" = B3 vs B2.
   * S1 = P(e) − P(c) demotes a prohibiting gold whenever the hypothesis is a positive question. The dev-selected
     bge-m3 B3 is S1 + lin-0.25. The one "found" TaG difference for H2 (bge-m3 Ru, +0.29) therefore follows from the
     score design and is not evidence about NLI as such. The paper should say this. The "decl" control has the same
     problem: `as_statement` produces the affirmative statement, which a prohibiting gold contradicts.
3. **Abbreviations in Russian only.** 14 of 25 Ru Tax questions use НДС/ИПН/КПН. In `data/corpus/tax_ru.jsonl`, "ИПН"
   occurs in **0** chunks and "НДС" in 6. The Kz versions spell the terms out, because the brief asked for terms "as in
   the Kazakh code text" ("жеке табыс салығы"; e.g. kz_tax_001). On the 11 test pairs whose Ru question has an
   abbreviation, bge-m3 B0 scores Ru 0.64 vs Kz 0.86; on the other 28 pairs it scores 0.96 vs 1.00. BM25-morph scores
   Ru 0.41 vs Kz 0.55 on the abbreviated pairs.
4. **Lexical overlap.** I recomputed the share of question lemmas present in the gold with the project's BM25
   analyzers: Ru general 0.638, Ru negation 0.537, Kz general 0.736, Kz negation 0.702. These match the paper's
   "70–74% vs 54–64%", but no script in the repo produces these numbers.

**Why it matters.** Points 3 and 4 make every Ru-vs-Kz comparison (H3, the D2 discussion "bge-m3 retrieves Kazakh
negation questions better than Russian ones", the e5-base H3 effect) confounded by how the questions were written,
not only by the language.

**What to do.** Report H3 and Ru-vs-Kz numbers with and without the abbreviated pairs. Add an abbreviation-expanded
Ru query condition (cheap, see §5). State the translation protocol in the paper.

### M7. A bag-of-words NEG token cannot carry predicate polarity, so the H1 null is predictable

**Evidence.** From the cached tokenized corpora (`data/index/tokens_morph_{ru,kz}.jsonl`):
* «не» occurs in 13,804 of 30,766 Ru chunks (44.9%, BM25 IDF 0.80);
* `NEG` occurs in 13,082 Kz chunks (42.5%, IDF 0.86);
* typical content words have IDF 4–8.

The negation cue also occurs in 22/24 negation golds, 16/25 general golds and 7/19 twins (Ru, rule regex). A token
that is unbound to its predicate and present in nearly half of the documents cannot separate "облагается" from
"не облагается" on the same topic.

**Why it matters.** The paper explains the null result of B2-noneg only by questions rarely containing negation. Even
negated questions would gain little from this design.

**What to do.** State the IDF argument. If time permits, test a scoped variant in which negation is bound to the next
content lemma ("не_облагаться", `NEG_бер`). This is CPU-only: re-tokenize and re-index.

### M8. The practitioner takeaway was not tested as stated and is not supported for bge-m3

**What.** The paper (Discussion: "a cross-encoder reranker over the dense top 30") and the deck (`takeaway.html`: "Put
a cross-encoder over the dense top 30") recommend reranking dense output. Every B3′ run reranks the top 30 of **B2**,
the hybrid (`run_ablation.py`: `top_k(s)` over `b2[key]`).

**Evidence.** I ran `paired_diff` on the stored per-query results, B3′ vs B0. For bge-m3 no difference is detected in
any cell: Ru negation +0.05 [0.00, 0.16], Ru TaG −0.07 [−0.23, 0.00], Kz negation −0.05 [−0.16, 0.00]. Differences
are found only for e5-base Kz: negation +0.37 (jina) and +0.42 (bge-reranker), TaG −0.36 [−0.62, −0.12]. "Removes most
of the rest" means bge-m3 polarity errors went from 3 to 1 (Ru 2 → 1, Kz 1 → 0).

**What to do.** Either run B3′ over the B0 top 30 (minutes of GPU) or reword the claim to "over the hybrid top 30".
Say that for bge-m3 no effect was detected and that the benefit was shown only for the weaker retriever.

### M9. The B3 vs B3′ comparison is not like-for-like

* **Input length.** The NLI classifier truncates premise + hypothesis at 256 tokens (`nli_eval.py: MAX_LEN = 256`) and
  was trained at 128 (`nli_train.py: MAX_LEN = 128`). The cross-encoders use 512 (`cross_encoder.py`). With the bge-m3
  tokenizer, 16 of 66 Ru and 18 of 66 Kz gold/twin chunks exceed 256 tokens, and 4 Ru and 3 Kz exceed 512 (largest:
  labor:99:2, 593 / 672 tokens).
* **Dev selection is noise.** The dev mean Recall@5 has a granularity of 0.05. For bge-m3 the selected s1-lin-0.25
  (dev 0.95) has test TaG 0.214, while s2-lin-0.25/0.5 (dev 0.90) have test TaG 0.107. The two dense models received
  different B3 variants.
* **The H2 conclusion still holds.** Even the best B3 variant chosen on test has mean Recall@5 0.871 (bge-m3) and
  0.825 (e5-base), below bge-reranker at 0.928 and 0.935. This is worth stating: it makes the H2 conclusion robust to
  the selection problem.

**What to do.** Report the full B3 grid on test as a sensitivity table, and rerun B3 at 512 tokens (GPU, < 1 h).

### M10. The NLI results mix denominators and label conventions

* **H6 compares different denominators.** The paper says: "variant c … calls 60.9% of the 110 range and numeric-bound
  entailments contradictions, against 0.9% for a. The base model … makes this error in 0% of the cases". The 60.9%
  and 0.9% come from `nli_compare.py` (`RANGE_KINDS = {"range", "number"}`, n = 110 = 67 range + 43 number
  entailments). The base model's 0% comes from `nli_eval.py`, which counts `kind == "range"` only (n = 67). The base
  model was never scored on the 110 items, and on `number` pairs its accuracy is 0.03 (`nli_eval_final-mDeBERTa…`,
  `by_kind.number`). On the same 67-item basis the figures are base 0%, a 1.5%, c 100%.
* **"Fine-tuning lowers accuracy on hand-written pairs"** comes with no interval: `nli_compare.py` never compares the
  base model. It is a 3–4 item difference on a set with ≥ 3 doubtful labels (M4).
* **Label conventions differ between the synthetic and manual sets.** The synthetic `number` neutrals label a tighter
  bound as neutral, e.g. «срок не более тридцати суток» → «не более четырех суток» (`data/nli/synthetic_ru.jsonl`). The
  manual set reads norms deontically: v002 "normal hours may be 45" is labeled contradiction, and so is v003. Under the
  deontic reading, a different statutory limit contradicts the norm. This mismatch may explain part of the drop on
  hand-written pairs, and it should be discussed.
* The Kz synthetic set has **no** `range` entailments at all (Counter over `synthetic_kz.jsonl`), so Kz H6 on
  synthetic data rests on 26 `number` items.

### M11. Claims differ between the paper, the deck and the decision log, and some are too strong

* D1: the deck (`hypotheses.html`) says "Held for all 6". The decision log of 16.11 says D1 is reported with the
  size-matched control as the main number, and under that control 5 of 6 encoders (all 6 in Russian) separate
  negation from a synonym edit. "Held" is not consistent with the authors' own decision.
* H5: the deck says "Failed". The decision log of 16.11 says "считается непроверяемой" (untestable), and the paper
  says "the comparison is weak". Use one status throughout.
* "The setup reproduces published numbers" (paper §6.1): only jina on NevIR is a like-for-like reproduction. The
  RuLawRetrieval comparison sets bge-m3 against e5-large and e5-base against e5-small. Call it a sanity check. I could
  not verify the dataset-card figures (0.820 / 0.710 / 0.480).
* "Neither helps" (abstract): B2 does help e5-base on Kazakh by the paper's own criterion. The abstract later says
  so, which makes it inconsistent with its own opening.

### M12. (Paper submission) Related work and citations of the resources used are missing

The paper uses XNLI, Kardeş-NLU, KazQAD, RuLawRetrieval, Russian SuperGLUE (RCB, LiDiRus), NevIR, bge-m3,
multilingual-e5, LaBSE, Qwen3-Embedding, EmbeddingGemma, mDeBERTa-v3 and the MoritzLaurer NLI checkpoint,
jina-reranker-v2, bge-reranker-v2-m3, pymorphy3, apertium-kaz, kaznlp, kazsearch, RRF and BM25, and cites none of
them. The project rule of citing only PDFs that were read explains the thin bibliography, but a venue will require
these citations. Relevant prior work the authors should look at (all to verify):
* NevIR (Weller et al., EACL 2024);
* ExcluIR (Zhang et al., 2024);
* "This is not a dataset" (García-Ferrero et al., EMNLP 2023);
* CONDAQA (Ravichander et al., EMNLP 2022);
* Hossain et al. (2020) on negation in NLI benchmarks;
* Kassner & Schütze (2020) on negated probes;
* BSARD statutory article retrieval (Louis & Spanakis, ACL 2022);
* Cormack et al. (2009) for RRF.

---

## 3. Minor issues

1. **The "2 of 24" count cannot be reproduced.** Paper (H2) and deck say "only 2 of the 24 Russian negation questions
   contain a negation word". The rule's own regex finds 1 (ru_lab_009). The BM25 analyzer's negation tokens appear in 4:
   ru_lab_004 «без», ru_lab_005 «нет», ru_lab_009 «не», ru_tax_009 «необлагаемого». Kz: 3 (kz_lab_005, kz_lab_009,
   kz_tax_009). State the definition.
2. **"On dev every α > 0 lowered recall" is inaccurate.** α = 0.002 ties α = 0 on dev for both models (bge-m3 0.90 vs
   0.90, e5-base 0.80 vs 0.80, `final_ablation_dev.json`). α = 0 won only on the TaG tie-break. The α grid is also
   coarse relative to RRF score gaps: about 2.6e-4 between ranks 1 and 2, so α = 0.002 shifts a document about 7 ranks.
3. **The fusion-weight choice is noise.** On dev, bge-m3 w = 0.25 and w = 2 tie at mean R@5 0.95
   (`controls_dev.json`) and were separated only by TaG. These weights point in opposite directions. The selection is
   not scripted. Present it as sensitivity only; the paper already partly does.
4. **Dev/test items are not independent.** ru_tax_001 (test) and ru_tax_002 (dev) share gold tax:382:5. ru_tax_003 and
   ru_tax_011 share twin tax:372:3, which is also a gold of ru_tax_025. The bootstrap treats correlated questions as
   independent.
5. **"Recall@5" is the gold share, not a hit.** For the two multi-gold questions (ru_tax_004, ru_tax_025) it is the
   share of gold found. Use hit@5 (any gold) or say clearly that it is gold recall.
6. **RRF tie handling.** Without `depth`, the many documents with zero BM25 score get BM25 ranks by corpus index order.
   Files are loaded alphabetically, so Tax is second to last and its zero-score chunks get the worst tie ranks. I
   measured that a median of 79% (Ru) and 57% (Kz) of the corpus scores zero per query. No effect on reported numbers:
   B2 equals B2-w1 with depth 1000 in every cell of `controls_test.json`. Still, the main B2 should use depth or treat
   zero scores as unranked.
7. **Truncation.** Dense and cross-encoder inputs are capped at 512 tokens (`dense.py: MAX_LEN`), although bge-m3
   handles 8k. 307 Ru and 225 Kz chunks exceed 300 words, with a maximum of 2,863 words. This contradicts "≤ 300 words"
   in PROGRESS.md (line 172) and the comment in `dense.py`. Report it, or raise the cap for bge-m3.
8. **D2 Russian control in the pool is broken.** `d2_affix.py::analyse_ru` tests `pieces[0] == "▁не"`, which only
   works for SentencePiece vocabularies with the `▁` prefix. In `results/d2_pool.json`, mBERT, ruBERT, LaBSE, Qwen3,
   Gemma and the Kazakh tokenizers all get `particle_ne_own_token_rate = 0.0`, which is impossible for WordPiece.
   These numbers are not in the paper, but the file is released.
9. **D2 construct validity.** Isolating the affix is neither necessary nor sufficient for a contextual encoder to
   encode negation. Kazakh monolingual tokenizers score 0% because frequent negative forms are single tokens. The 70%
   threshold is arbitrary. The paper's sentence "does not translate into worse Kazakh retrieval" rests on one model and
   is confounded by M6.3–4. Keep D2 descriptive.
10. **D1 is measured at the wrong level.** D1 compares passages with passages under artificial edits. The
    retrieval-relevant analog, cos(q, gold) − cos(q, twin) over real twins, is not computed. Cosine magnitudes are not
    comparable across models (anisotropy). PROGRESS has a scale-free "negation position" that the paper drops.
11. **EmbeddingGemma wording.** The paper and the deck say "positive for five of the six encoders in both languages
    … EmbeddingGemma is the exception". In Russian, EmbeddingGemma's control margin is +0.005 [0.001, 0.009], positive
    with the CI above 0 (`d1_pool_ru.json`). Only Kz (−0.003 [−0.007, 0.001]) is the exception.
12. **H5 reporting is selective.** The paper omits the one significant H5 comparison, b vs a on manual-ru,
    +0.08 [+0.02, +0.16] (`nli_h5_h6_2026-11-16.json`).
13. **H3 headroom.** The e5-base Kz baseline (0.47) has more room to improve than Ru (0.74). A larger gain on Kz partly
    reflects headroom. Normalize the gain or discuss it.
14. **Error analysis by the agent.** The 20 cases and their categories were selected and assigned by the LLM agent
    (`data/eval/error_notes.tsv`), and the paper does not say so. One case (ru_tax_006) shows that the twins are
    incomplete, so TaG is a lower bound. One case (ru_tax_014) shows the gold is not exhaustive.
15. **Unreported metrics.** The planned NevIR-style pairwise accuracy (CLAUDE.md §6) was not computed. MRR@10 and R@10
    are computed but not reported.
16. **Averaged comparison.** "Base NLI equals fine-tuned" is an average over 10 scoring/fusion variants. That average
    includes the `pure` variants, which are known to collapse (base 0.799 vs tuned 0.809 for bge-m3, recomputed from
    `controls_test.json`). Compare the dev-selected or best variants with a paired test instead.
17. **Corpus release.** The 7 rejected alignment pairs stay in the corpus under shared ids with different content,
    e.g. labor:26:1/26:2 swapped. Flag or fix them in the release. `data/corpus/alignment_review.json` marks labor:178
    as "no", while PROGRESS.md says it "counts as aligned".
18. **Unchecked Kz synthetic data.** The Kz synthetic NLI data (NEEDS_KZ_CHECK) was used in training without a check.
    This is acknowledged, but it conflicts with CLAUDE.md §0.3 ("do not use in metrics until checked"); training is
    arguably not "metrics", but say so.
19. **Missing paper content.** A table of hypotheses with their pre-set refutation criteria (CLAUDE.md §3), dataset
    statistics, Kazakh question examples, hardware and runtime, software versions, the annotation protocol (interface,
    label visibility, timing) and the data license.

---

## 4. Bugs and inconsistencies found

| # | Where | What | Effect |
|---|---|---|---|
| B1 | `data/diagnostics/d1_triples.jsonl` kz001, kz013, kz021, kz030 | `min_paraphrase` is **identical** to `sentence` (cos = 1.0 in `d1_margin_final_kz.json`). Introduced by the annotator's edits (`data/review/answers/reviews/d1-kz001.json` etc.); the original drafts had «құқығы бар». | The Kz control margin is inflated on 4/30 items. Recomputed without them: e5-base 0.0253 → 0.0252, bge-m3 0.0970 → 0.0920, Qwen3 0.0256 → 0.0231, Gemma −0.0031 → −0.0042. Conclusion unchanged; n should be 26. |
| B2 | Paper §NLI/H6 | 60.9% / 0.9% (n = 110, range + number) placed next to the base model's 0% (n = 67, range only) | Misleading comparison (M10) |
| B3 | `src/diagnostics/d2_affix.py::analyse_ru` | `"▁не"` check is valid only for SentencePiece | Invalid Ru control for 10 of 14 tokenizers in `results/d2_pool.json` |
| B4 | `data/nli/val_manual.jsonl` v019, v030, v035 | Annotator changes C→E, C→E, N→E appear wrong | About 6% of labels, both languages (M4) |
| B5 | Paper H2, deck `negtoken.html` | "2 of 24 negation questions contain a negation word" | 1 (rule regex) or 4 (BM25 analyzer) |
| B6 | Paper H2 | "every α > 0 lowered recall" on dev | α = 0.002 tied |
| B7 | Paper D1, deck `d1.html` | "EmbeddingGemma is the exception" in both languages | Only in Kazakh |
| B8 | Paper Discussion, deck `takeaway.html` | "cross-encoder over the dense top 30" | Runs rerank the B2 top 30 (M8) |
| B9 | Deck `hypotheses.html` | D1 "Held for all 6", H5 "Failed" | Conflicts with the decision log of 16.11 |
| B10 | PROGRESS.md line 172, `dense.py` comment | chunks "≤ 300 words" | 307 Ru chunks > 300 words, max 2,863 |
| B11 | `alignment_review.json` vs PROGRESS.md | labor:178 "no" vs "counts as aligned" | Documentation only |
| B12 | `fusion.py::rrf` without depth | zero-score BM25 docs ranked by corpus order | None observed (B2 = B2-w1) |
| B13 | `nli_eval.py` vs `nli_compare.py` | different definitions of range-negation error (`range` vs `range` + `number`) | Source of B2 |

**Numbers I checked and found consistent.**
* Paper Table 2 (all 48 cells) matches `final_table.json` / `final_table.md`.
* Table 1 (D1) matches `d1_pool_{ru,kz}.json`.
* Table 3 (NLI) matches `nli_eval_final-*.json`.
* All H1/H2/H3 intervals quoted in the paper match `final_table.json["hypothesis_tests"]`.
* Control-run numbers match `controls_test.json`.
* The NLI-control averages (0.80/0.81, 0.74/0.76, TaG 0.21/0.20, 0.24/0.24, decl 0.70) are reproduced.
* D2: 21.1%, 95.9%, 0–49.7%.
* Alignment: 99.7% / 99.4%.
* Error-analysis ranks (VAT 1 → 82 → 6, ИПН 1 → 70 → 9, tax_001 gold 99 / twin 5).
* The deck numbers on `ladder.html`, `twins.html`, `fusion.html`, `finetune.html`, `nli.html`, `abbrev.html`,
  `problem.html` and `setup.html`.
* NLI training sizes: 105,986 / 55,988 pairs.

---

## 5. Missing experiments and analyses

### Cheap (hours, no new annotation, mostly CPU or seconds of GPU)
1. Exact paired tests (McNemar/sign, permutation) with discordant counts and Holm/BH for the H1–H3 family (M2).
2. Wilson intervals for absolute TaG and Recall@5. Add TaG@5/@10 and the "polarity-induced miss" (M1, M5).
3. Breakdown by code (Labor vs Tax) and by abbreviated vs full-term questions (M6.3).
4. Ru queries with abbreviations expanded (НДС → «налог на добавленную стоимость», ИПН, КПН) for BM25, B0 and B2.
   BM25 needs CPU only; dense needs about 50 query encodings.
5. B3′ over the **B0** top 30, to test the practitioner claim (M8).
6. The full B3 grid on test as a sensitivity table; note that even the test-oracle B3 is below B3′ (M9).
7. D1 Kz with the 4 degenerate controls removed (B1). Embeddings for 30 sentences take seconds.
8. Manual NLI accuracies with v019, v030 and v035 corrected or excluded, plus base vs a paired tests (M4, M10).
9. Base model's range-negation error on the same 110 synthetic items (M10).
10. A scoped negation token for BM25 (`не_<lemma>`, `NEG_<stem>`), plus the IDF table (M7).
11. Fix or remove the Ru control in `d2_pool.json` (B3); a script for the overlap numbers in M6.4.
12. Pin versions (`pip freeze`), record model revisions (HF commit hashes), and the commit hashes of
    `../nlp_tools/{kazsearch-py, kaznlp, apertium-kaz}`. Extend the README to `finalize.py` and the full reproduction
    path.

### Medium (needs a GPU or about a day)
1. B3 at 512 tokens, training at a matching length (M9).
2. The retrieval-level analog of D1: cos(q, gold) − cos(q, twin) over the 14 twin questions, against the rank gaps.
3. bge-m3's own sparse and multi-vector (ColBERT) modes as the lexical component, instead of external BM25.
4. Multi-query retrieval with both polarity statements (q⁺ "X is allowed", q⁻ "X is prohibited"), then union or a
   cross-encoder over them.
5. A cross-encoder fine-tuned on legal contrastive pairs, with twins or polarity-flipped distractor norms from the 20
   distractor codes as hard negatives. NevIR repro reports cross-encoders benefit most from such tuning.
6. 2–3 NLI seeds for a and c (2.6 h each, already scripted in `src/rerank/nli_seeds.py`).
7. An LLM listwise reranker as an upper-bound reference, only if within scope. The plan excludes generation, but a
   reranker is retrieval.

### Expensive (needs new annotation)
1. Question set v2 with balanced cells (`data/eval/queries_v2_draft.jsonl`, 69 drafts exist), frozen before any
   method change.
2. A blind second annotator with Cohen's κ on questions, gold/twin and NLI labels (page already built).
3. Pooled relevance judgments for the top 5 of every system, to see multiple golds and missing twins.
4. Contrastive question pairs: for each twin, the opposite question whose gold is the twin. This is the NevIR-style
   design and the most direct test of polarity sensitivity; it also enables the planned pairwise accuracy.
5. A human check of the Kz synthetic NLI data and of the deontic label convention (M10).

---

## 6. Recommendations, prioritised

### Before the defense on 1 Dec 2026 (no new annotation)
1. **Restate the claims (M1, M2, M8, M11).** Make the five "found" effects suggestive and give their discordant
   counts and exact p-values. Change the takeaway to "a cross-encoder over the hybrid top 30 helped the weaker
   retriever; for bge-m3 no effect was detected". Remove "Held" for D1 in favour of the controlled result. Use one H5
   status. Make "reproduces" "sanity check".
2. **Disclose the protocol (M3, M4).** Questions identical to LLM drafts (98/98), 100% acceptance, about 17 s per
   card, the annotator saw the labels, B3 fusion developed on the same questions, gold of tax_001/002 revised from
   retrieval output. One slide or paragraph is enough, and examiners will value it.
3. **Fix the small data and text bugs.** B1 (report D1 Kz on n = 26), B2, B5, B6, B7, B10, and flag v019, v030 and
   v035 (B4).
4. **Add the cheap analyses with the highest payoff.** Cheap items 1–3 (tests, intervals, TaG@k, by code and
   abbreviation), then 5 (B3′ over B0) and 4 (abbreviation expansion). Each is under an hour, and together they turn
   several weak claims into precise ones.
5. **Prepare for examiner questions.** "Why does a NEG token in 42% of documents help at all?" (M7). "What does your
   TaG measure for e5-base Kz?" (M5).

### Before submitting a paper
1. Build the v2 question set as a frozen, balanced test set with a blind second annotator, κ, pooled judgments and
   contrastive question pairs (Expensive 1–4). Without it the retrieval conclusions remain a pilot.
2. Run like-for-like B3 (512 tokens), the full grid reported, ≥ 3 seeds, and S2 or a question-aware score instead of
   S1 for yes/no questions.
3. Write a related-work section and cite every dataset, model and tool used (M12).
4. Reframe the contribution around the parallel statutory corpus and the twin construct, and treat the
   negative-component results as secondary.
5. Release the alignment-rejected chunks as such, pin the environment, and give a one-command reproduction.

---

## 7. Questions to the authors

1. How much time did the annotator actually spend per question card? Did she read the gold and twin texts, or only
   the question wording? Was the "Следует" (entailment) button perhaps understood as "accept the draft"? That would
   explain v019, v030 and v035.
2. Why did the annotator set the Kz minimal paraphrase equal to the original sentence in kz001, kz013, kz021 and kz030?
   Was «құқығы бар» judged unacceptable, and if so, what should the control be?
3. Were the B3 fusion modes and λ grid fixed before or after looking at the per-question draft results of 15.11? Did
   anyone look at which questions B3-pure lost?
4. Was any retriever used to find the gold and twin candidates when drafting (`make_draft_ru.py` says "found by
   searching the corpus")? Which one?
5. Why was S1 kept in the dev grid for yes/no questions whose correct answer is usually a prohibition? Was a
   question-aware score considered, e.g. max(P(e), P(c)), or NLI against both q⁺ and q⁻?
6. Were the Ru tax questions written with abbreviations on purpose, to mimic real users? If so, why do the Kz versions
   expand them?
7. How was "2 of 24 negation questions contain a negation word" counted?
8. For the takeaway, do you have any run with a cross-encoder over B0 instead of B2?
9. Why is the NLI classifier capped at 256 tokens while the cross-encoders get 512?
10. Under which reading are the synthetic "tighter numeric bound" pairs neutral, given that the manual set labels a
    different statutory limit as a contradiction (v002, v003)?
11. Is there a reason not to report MRR@10 and R@10, which are computed?
12. Which redistribution terms apply to the adilet texts, and to kazsearch (LGPL-3.0) if parts of it are bundled?

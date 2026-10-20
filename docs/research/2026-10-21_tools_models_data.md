# Research 21.10.2026: tools, models, data, related work

Four web-research passes (subagents), each item opened on HF / GitHub / arXiv / publisher. Marks: **[F]** full
text or files checked, **[A]** abstract / card only. Items checked by hand in the main session are marked ✔.
Nothing here is adopted yet; decisions go to CLAUDE.md §12.

## Related work: positioning
* **No multilingual NevIR and no Ru/Kz negation retrieval benchmark found** — the gap we fill.
* Our metrics are adaptations, to be framed as such:
  Twin-above-Gold ← NevIR pairwise accuracy (Weller et al., EACL 2024, https://aclanthology.org/2024.eacl-long.139/);
  Negation Margin ≈ the "negation blindness" diagnostic of Cao (ECAI 2025, https://arxiv.org/abs/2504.00584);
  Affix Isolation Rate = MorphScore restricted to one morpheme (Arnett et al. 2025, 70 languages incl. Kazakh,
  https://arxiv.org/abs/2507.06378 [F]; Kazakh boundary precision 0.15–0.46; alignment explains ~2.4% of downstream
  variance on average — our point: for one semantically critical affix it matters).
* ✔ **Kalzhanov et al. 2026**, *Information* 17(10):1042 (19.10.2026), https://www.mdpi.com/2078-2489/17/10/1042 —
  41 RK acts, article level, Ru+Kz, 150 lawyer-verified QA, BM25 / dense / agent. Focus: dates, repealed norms,
  citation errors; finding "retrieval, not generation, causes most citation errors" supports our thesis.
  No negation/polarity. Must cite; check corpus licence.
* NLI filtering of retrieved passages: Yoran et al., ICLR 2024 (https://arxiv.org/abs/2310.01558) — helps but drops
  relevant passages → report the nDCG cost of our NLI reranker. Contradiction retrieval: SparseCL (ICML 2025).
* Affixal negation: Tan & Blanco, EMNLP 2026 (https://arxiv.org/abs/2609.13685) [A] — English? check full text.
* Deontic modality: LEXDEMOD (EMNLP 2022); deontological keyword bias of LLMs (ACL 2025).
* Kazakh legal AI: Rakhimova et al. 2025 (MDPI Computers 14(9):354) QA on Adilet, no retrieval/negation.

## Models (for the widened pool)
| model | params | tokenizer | note |
|---|---|---|---|
| intfloat/multilingual-e5-base (current) | 278M | XLM-R SP 250k | KazMTEB-v0 60.4 (author-reported table) |
| intfloat/multilingual-e5-large-instruct | 560M | XLM-R SP | KazMTEB-v0 70.5; same tokenizer → isolates size/instructions |
| BAAI/bge-m3 | 568M | XLM-R SP | best Kz retrieval in the tables seen; 8k context |
| sentence-transformers/LaBSE | 471M | WordPiece 501k | bitext-trained |
| Qwen/Qwen3-Embedding-0.6B | 596M | byte-level BPE 151k | decoder architecture |
| google/embeddinggemma-2 | ~740M | Gemma BPE 262k | released 06.10.2026, needs recent transformers |
Rerankers: jina-reranker-v2 (NevIR 65.2%, CC-BY-NC), BAAI/bge-reranker-v2-m3 (NevIR 43.5%), Qwen3-Reranker-0.6B.
Zero-shot NLI: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (ru yes, kk no), joeddav/xlm-roberta-large-xnli.
Extra tokenizers for D2 only: mBERT, mDeBERTa, ruBERT, kz-transformers/kaz-roberta-conversational, HPLT kk BERT,
Eraly-ml/KazBERT, **Abzalbek89/kk-bert-small-{bpe, morph-bpe, unigram}** (same corpus, three tokenizers — a controlled
comparison), deepvk/USER-bge-m3 (pruned Russian vocab: check Kazakh letters survive).

## Tools
* **kazsearch-py** (github.com/iDynbek/kazsearch-py, LGPL-3.0): Kazakh BM25 stemmer with an explicit negation layer
  (`LAYER_VNEG`). Out of the box it strips the negation (төлемейді → төле) — keep the layer as a `NEG` token.
* **Stanza kk** (Apache-2.0, UD Kazakh-KTB `Polarity=Neg`): third, neural, independent negation checker.
* **Stop-word trap:** NLTK `russian` contains не/нет/ни/без, NLTK `kazakh` contains жоқ/емес — whitelist them.
* Russian «не-»: pymorphy3 prefix analysis only fires for unknown words; split «не» when the rest is a known word
  of the same POS (exceptions небо, нефть, недо-…); Slovorez (morpheme segmentation, ONNX, MIT) as a check.
* FEMSeg_kaz_v3 (github.com/walsher46/Kazakh-tokenizer, MIT) — neural segmenter; ending table useful for regex coverage.
* KazStem (apertium-based, Linux/WSL) — not independent of apertium.

## Data
* **Kardeş-NLU** (EACL 2024, github.com/lksenel/Kardes-NLU): Kazakh XNLI **dev 1000 / test 2000, MT post-edited by a
  native speaker** — the missing Kazakh NLI validation. No licence file: research use with citation.
* sultanbi/snli-kazakh: 549k MT Kazakh SNLI (CC BY-SA) — silver kk NLI training, filter by negation preservation.
* boun-tabi/nli_tr (SNLI-TR, MNLI-TR, MT): more volume than XNLI-tr, not better quality.
* Russian negation diagnostics: RCB (`negation` field), LiDiRus (~108 negation items) in russian_super_glue (MIT).
* Retrieval sanity: issai/kazqad-retrieval test (1 929 native questions, gated); alekseevpavel04/ru-law-retrieval
  `tk_hard` (716, Russian Labor Code, LLM-generated questions).
* bobur-m/adiletcodex (CC BY 4.0): 65 807 articles of 343 acts Ru/Kz/En — independent check of our Labor Code
  (227/227); its Tax Code is the old 2017 one.
* raiym76/kazlawbench (CC BY-NC-SA, LLM-synthesized), olzhasAl/adilet-legal-qa-kz (no Labor/Tax) — style reference only.
* issai/kazparc: human-translated kk/ru/en/tr, ~20.8% legal domain; gated, licence not stated.
* Negation (English): NevIR, ExcluIR, CONDAQA, This-is-not-a-dataset, jinaai/negation-dataset.

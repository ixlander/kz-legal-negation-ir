# Exact tests and intervals (test split)

## Pre-registered comparisons, exact sign test, Holm over all 40

| comparison | better / worse / n | p | p (Holm) |
|---|---|---|---|
| H2 e5-base kz B3 vs B3'-bge-reranker-v2-m3 negation hit@5 | 0 / 5 / 19 | 0.062 | 1.00 |
| H3 e5-base B2 kz-ru gain negation | 5 / 0 / 19 | 0.062 | 1.00 |
| H2 bge-m3 ru B3 vs B3'-bge-reranker-v2-m3 twin-above-gold | 4 / 0 / 14 | 0.125 | 1.00 |
| H1 e5-base kz B2 vs B0 negation hit@5 | 4 / 0 / 19 | 0.125 | 1.00 |
| H2 e5-base kz B3 vs B3'-jina-v2 negation hit@5 | 0 / 4 / 19 | 0.125 | 1.00 |
| H2 bge-m3 ru B3 vs B3'-jina-v2 twin-above-gold | 5 / 1 / 14 | 0.219 | 1.00 |
| H3 e5-base B2 kz-ru gain general | 5 / 2 / 20 | 0.453 | 1.00 |
| H2 bge-m3 ru B3 vs B3'-bge-reranker-v2-m3 negation hit@5 | 0 / 2 / 19 | 0.500 | 1.00 |
| H2 bge-m3 ru B3 vs B2 twin-above-gold | 2 / 0 / 14 | 0.500 | 1.00 |
| H1 bge-m3 kz B2 vs B0 negation hit@5 | 0 / 2 / 19 | 0.500 | 1.00 |
| H1 e5-base ru B2 vs B0 general hit@5 | 0 / 2 / 20 | 0.500 | 1.00 |
| H2 e5-base ru B3 vs B3'-jina-v2 twin-above-gold | 2 / 0 / 14 | 0.500 | 1.00 |
| H2 e5-base ru B3 vs B3'-bge-reranker-v2-m3 negation hit@5 | 0 / 2 / 19 | 0.500 | 1.00 |
| H2 e5-base ru B3 vs B2 negation hit@5 | 2 / 0 / 19 | 0.500 | 1.00 |
| H2 e5-base kz B3 vs B3'-bge-reranker-v2-m3 twin-above-gold | 2 / 0 / 14 | 0.500 | 1.00 |
| H3 e5-base B3 kz-ru gain negation | 2 / 0 / 19 | 0.500 | 1.00 |
| H1 bge-m3 ru B2 vs B0 general hit@5 | 1 / 3 / 20 | 0.625 | 1.00 |
| H2 bge-m3 ru B3 vs B3'-jina-v2 negation hit@5 | 1 / 3 / 19 | 0.625 | 1.00 |
| H2 e5-base ru B3 vs B3'-bge-reranker-v2-m3 twin-above-gold | 3 / 1 / 14 | 0.625 | 1.00 |
| H1 e5-base kz B2 vs B0 general hit@5 | 4 / 2 / 20 | 0.688 | 1.00 |
| H1 bge-m3 ru B2 vs B0 negation hit@5 | 0 / 1 / 19 | 1.000 | 1.00 |
| H1 bge-m3 kz B2 vs B0 general hit@5 | 0 / 1 / 20 | 1.000 | 1.00 |
| H2 bge-m3 kz B3 vs B3'-jina-v2 negation hit@5 | 1 / 2 / 19 | 1.000 | 1.00 |
| H2 bge-m3 kz B3 vs B3'-jina-v2 twin-above-gold | 1 / 1 / 14 | 1.000 | 1.00 |
| H2 bge-m3 kz B3 vs B3'-bge-reranker-v2-m3 negation hit@5 | 0 / 1 / 19 | 1.000 | 1.00 |
| H2 bge-m3 kz B3 vs B3'-bge-reranker-v2-m3 twin-above-gold | 1 / 0 / 14 | 1.000 | 1.00 |
| H3 bge-m3 B2 kz-ru gain negation | 1 / 2 / 19 | 1.000 | 1.00 |
| H3 bge-m3 B2 kz-ru gain general | 3 / 2 / 20 | 1.000 | 1.00 |
| H3 bge-m3 B3 kz-ru gain negation | 1 / 2 / 19 | 1.000 | 1.00 |
| H3 bge-m3 B3 kz-ru gain general | 2 / 2 / 20 | 1.000 | 1.00 |
| H1 e5-base ru B2 vs B0 negation hit@5 | 0 / 1 / 19 | 1.000 | 1.00 |
| H2 e5-base ru B3 vs B3'-jina-v2 negation hit@5 | 1 / 2 / 19 | 1.000 | 1.00 |
| H2 e5-base ru B3 vs B2 twin-above-gold | 1 / 1 / 14 | 1.000 | 1.00 |
| H2 e5-base kz B3 vs B3'-jina-v2 twin-above-gold | 2 / 1 / 14 | 1.000 | 1.00 |
| H2 e5-base kz B3 vs B2 negation hit@5 | 1 / 2 / 19 | 1.000 | 1.00 |
| H2 e5-base kz B3 vs B2 twin-above-gold | 2 / 1 / 14 | 1.000 | 1.00 |
| H3 e5-base B3 kz-ru gain general | 3 / 2 / 20 | 1.000 | 1.00 |

Comparisons with no changed question are omitted.

## Twin-above-Gold, full ranking vs top 5 (14 questions with a twin)

| cell | TaG k/n [Wilson] | TaG@5 | polarity miss@5 |
|---|---|---|---|
| bge-m3 · ru · B0 | 2/14 [0.04, 0.399] | 2/14 | 1/14 |
| bge-m3 · kz · B0 | 1/14 [0.013, 0.315] | 1/14 | 0/14 |
| bge-m3 · ru · B2 | 3/14 [0.076, 0.476] | 3/14 | 2/14 |
| bge-m3 · kz · B2 | 1/14 [0.013, 0.315] | 1/14 | 1/14 |
| bge-m3 · ru · B3'-jina-v2 | 1/14 [0.013, 0.315] | 1/14 | 1/14 |
| bge-m3 · kz · B3'-jina-v2 | 1/14 [0.013, 0.315] | 1/14 | 0/14 |
| bge-m3 · ru · B3'-bge-reranker-v2-m3 | 1/14 [0.013, 0.315] | 1/14 | 1/14 |
| bge-m3 · kz · B3'-bge-reranker-v2-m3 | 0/14 [0.0, 0.215] | 0/14 | 0/14 |
| bge-m3 · ru · B3-nli-a-s1-lin-0.25 | 5/14 [0.163, 0.612] | 5/14 | 2/14 |
| bge-m3 · kz · B3-nli-a-s1-lin-0.25 | 1/14 [0.013, 0.315] | 0/14 | 0/14 |
| e5-base · ru · B0 | 4/14 [0.117, 0.546] | 2/14 | 0/14 |
| e5-base · kz · B0 | 6/14 [0.214, 0.674] | 1/14 | 0/14 |
| e5-base · ru · B2 | 3/14 [0.076, 0.476] | 2/14 | 1/14 |
| e5-base · kz · B2 | 2/14 [0.04, 0.399] | 0/14 | 0/14 |
| e5-base · ru · B3'-jina-v2 | 1/14 [0.013, 0.315] | 0/14 | 0/14 |
| e5-base · kz · B3'-jina-v2 | 2/14 [0.04, 0.399] | 1/14 | 0/14 |
| e5-base · ru · B3'-bge-reranker-v2-m3 | 1/14 [0.013, 0.315] | 1/14 | 0/14 |
| e5-base · kz · B3'-bge-reranker-v2-m3 | 1/14 [0.013, 0.315] | 0/14 | 0/14 |
| e5-base · ru · B3-nli-a-s2-rrf | 3/14 [0.076, 0.476] | 2/14 | 1/14 |
| e5-base · kz · B3-nli-a-s2-rrf | 3/14 [0.076, 0.476] | 2/14 | 2/14 |

## hit@5 by group

| cell | k/n | rate |
|---|---|---|
| bge-m3 · B0 · ru · labor | 18/18 | 1.00 |
| bge-m3 · B0 · ru · tax, Ru abbreviated | 7/11 | 0.64 |
| bge-m3 · B0 · ru · tax, other | 9/10 | 0.90 |
| bge-m3 · B0 · kz · labor | 18/18 | 1.00 |
| bge-m3 · B0 · kz · tax, Ru abbreviated | 10/11 | 0.91 |
| bge-m3 · B0 · kz · tax, other | 10/10 | 1.00 |
| bge-m3 · B2 · ru · labor | 17/18 | 0.94 |
| bge-m3 · B2 · ru · tax, Ru abbreviated | 6/11 | 0.55 |
| bge-m3 · B2 · ru · tax, other | 8/10 | 0.80 |
| bge-m3 · B2 · kz · labor | 18/18 | 1.00 |
| bge-m3 · B2 · kz · tax, Ru abbreviated | 8/11 | 0.73 |
| bge-m3 · B2 · kz · tax, other | 9/10 | 0.90 |
| bge-m3 · B3'-jina-v2 · ru · labor | 16/18 | 0.89 |
| bge-m3 · B3'-jina-v2 · ru · tax, Ru abbreviated | 10/11 | 0.91 |
| bge-m3 · B3'-jina-v2 · ru · tax, other | 9/10 | 0.90 |
| bge-m3 · B3'-jina-v2 · kz · labor | 17/18 | 0.94 |
| bge-m3 · B3'-jina-v2 · kz · tax, Ru abbreviated | 10/11 | 0.91 |
| bge-m3 · B3'-jina-v2 · kz · tax, other | 9/10 | 0.90 |
| bge-m3 · B3'-bge-reranker-v2-m3 · ru · labor | 18/18 | 1.00 |
| bge-m3 · B3'-bge-reranker-v2-m3 · ru · tax, Ru abbreviated | 8/11 | 0.73 |
| bge-m3 · B3'-bge-reranker-v2-m3 · ru · tax, other | 10/10 | 1.00 |
| bge-m3 · B3'-bge-reranker-v2-m3 · kz · labor | 18/18 | 1.00 |
| bge-m3 · B3'-bge-reranker-v2-m3 · kz · tax, Ru abbreviated | 9/11 | 0.82 |
| bge-m3 · B3'-bge-reranker-v2-m3 · kz · tax, other | 10/10 | 1.00 |
| e5-base · B0 · ru · labor | 18/18 | 1.00 |
| e5-base · B0 · ru · tax, Ru abbreviated | 8/11 | 0.73 |
| e5-base · B0 · ru · tax, other | 7/10 | 0.70 |
| e5-base · B0 · kz · labor | 15/18 | 0.83 |
| e5-base · B0 · kz · tax, Ru abbreviated | 5/11 | 0.46 |
| e5-base · B0 · kz · tax, other | 5/10 | 0.50 |
| e5-base · B2 · ru · labor | 17/18 | 0.94 |
| e5-base · B2 · ru · tax, Ru abbreviated | 6/11 | 0.55 |
| e5-base · B2 · ru · tax, other | 7/10 | 0.70 |
| e5-base · B2 · kz · labor | 15/18 | 0.83 |
| e5-base · B2 · kz · tax, Ru abbreviated | 8/11 | 0.73 |
| e5-base · B2 · kz · tax, other | 8/10 | 0.80 |
| e5-base · B3'-jina-v2 · ru · labor | 17/18 | 0.94 |
| e5-base · B3'-jina-v2 · ru · tax, Ru abbreviated | 10/11 | 0.91 |
| e5-base · B3'-jina-v2 · ru · tax, other | 9/10 | 0.90 |
| e5-base · B3'-jina-v2 · kz · labor | 17/18 | 0.94 |
| e5-base · B3'-jina-v2 · kz · tax, Ru abbreviated | 8/11 | 0.73 |
| e5-base · B3'-jina-v2 · kz · tax, other | 9/10 | 0.90 |
| e5-base · B3'-bge-reranker-v2-m3 · ru · labor | 18/18 | 1.00 |
| e5-base · B3'-bge-reranker-v2-m3 · ru · tax, Ru abbreviated | 9/11 | 0.82 |
| e5-base · B3'-bge-reranker-v2-m3 · ru · tax, other | 10/10 | 1.00 |
| e5-base · B3'-bge-reranker-v2-m3 · kz · labor | 18/18 | 1.00 |
| e5-base · B3'-bge-reranker-v2-m3 · kz · tax, Ru abbreviated | 8/11 | 0.73 |
| e5-base · B3'-bge-reranker-v2-m3 · kz · tax, other | 10/10 | 1.00 |
| bm25-morph · ru · labor | 14/18 | 0.78 |
| bm25-morph · ru · tax, Ru abbreviated | 5/11 | 0.46 |
| bm25-morph · ru · tax, other | 4/10 | 0.40 |
| bm25-morph · kz · labor | 17/18 | 0.94 |
| bm25-morph · kz · tax, Ru abbreviated | 6/11 | 0.55 |
| bm25-morph · kz · tax, other | 7/10 | 0.70 |

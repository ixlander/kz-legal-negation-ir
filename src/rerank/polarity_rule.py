"""Rule-based polarity booster (B3-rule): a deliberately simple baseline for the NLI reranker.

1. Query polarity is NEG when it contains a negation cue, else POS.
   Ru cues: «не», «нельзя», «запрещ…»; Kz cues: a negative verb form (NEG token), «емес», «жоқ».
2. Chunk polarity: the same rule on the chunk sentence with the largest lexical overlap
   with the query.
3. score = base_score + alpha if the polarities match, base_score - alpha otherwise.
alpha is tuned on the dev split.
"""
import re

from src.retrieval.bm25 import analyzer

RU_NEG = re.compile(r"(?<![а-яё])(не|нельзя|запрещ[а-яё]*)(?![а-яё])")
KZ_NEG_WORDS = {"емес", "жоқ", "NEG"}
SENTENCE = re.compile(r"(?<=[.;:!?])\s+|\n+")


def is_negative(text, lang, tokens=None):
    if lang == "ru":
        return bool(RU_NEG.search(text.lower().replace("ё", "е")))
    return bool(KZ_NEG_WORDS & set(tokens if tokens is not None else analyzer("morph", "kz")(text)))


def chunk_polarity(query_tokens, chunk, lang):
    tok = analyzer("morph", lang)
    q = set(query_tokens)
    best, best_overlap = chunk, -1
    for s in SENTENCE.split(chunk):
        if not s.strip():
            continue
        overlap = len(q & set(tok(s)))
        if overlap > best_overlap:
            best, best_overlap = s, overlap
    return is_negative(best, lang)


def boost(query, chunks, base_scores, lang, alpha):
    tok = analyzer("morph", lang)
    q_tokens = tok(query)
    q_neg = is_negative(query, lang, q_tokens)
    return [s + (alpha if chunk_polarity(q_tokens, c, lang) == q_neg else -alpha)
            for c, s in zip(chunks, base_scores)]

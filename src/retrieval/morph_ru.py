"""Russian analyzer for the morphology-aware BM25 (B2).

* lemmatization with pymorphy3;
* function words are dropped by part of speech (conjunctions, prepositions, particles,
  interjections) — except the negation whitelist, so «не вправе» keeps «не»;
* a fused negative prefix is split off: «неуплата» -> «не» + «уплата», only when the rest is
  a known word with the same part of speech (so «небо», «нефть», «невеста» stay whole).

No external stop-word list: the NLTK Russian list contains «не», «нет», «ни», «без»
(docs/research/2026-10-21_tools_models_data.md).
"""
import re
from functools import lru_cache

import pymorphy3

MORPH = pymorphy3.MorphAnalyzer()
NEG_WORDS = {"не", "ни", "нет", "без"}
DROP_POS = {"CONJ", "PREP", "PRCL", "INTJ"}
# «не»-initial lemmas where the rest happens to be a word of the same POS but there is no negation
# (checked on the Labor Code word list), plus «недо-» = "under-" words
NOT_PREFIX = {"невеста", "нерв", "нерпа", "нефть", "небо", "неделя", "недра", "некий", "несколько",
              "нечто", "некто", "негде", "некогда", "некуда", "незачем", "непосредственно",
              "непосредственный", "немедленно", "немедленный", "некоторый", "неотложный", "неотложно",
              "неприкосновенность", "неприкосновенный", "недостаток", "недостача", "недоимка",
              "недоработать", "недоработанный", "недовес", "недоплата", "необходимый"}
# predicatives that pymorphy3 tags differently from their positive form
FORCE_SPLIT = {"невозможно": "возможно", "недопустимо": "допустимо", "нецелесообразно": "целесообразно",
               "неправомерно": "правомерно", "незаконно": "законно"}
# parts of speech that pymorphy3 may assign differently to «X» and «неX» without a real change
POS_GROUP = {"ADJF": "A", "PRTF": "A", "ADJS": "S", "PRTS": "S", "ADVB": "D", "PRED": "D"}
WORD_RE = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*")


@lru_cache(maxsize=200_000)
def _parse(word):
    return MORPH.parse(word)[0]


@lru_cache(maxsize=200_000)
def split_ne(word):
    """'неуплата' -> 'уплата' if the fused «не-» is a negation prefix, else None."""
    if word in FORCE_SPLIT:
        return FORCE_SPLIT[word]
    if not word.startswith("не") or len(word) < 6:
        return None
    p = _parse(word)
    if p.normal_form in NOT_PREFIX or word in NOT_PREFIX:
        return None
    rest = word[2:]
    if not MORPH.word_is_known(rest):
        return None
    pos_w, pos_r = str(p.tag.POS), str(_parse(rest).tag.POS)
    return rest if POS_GROUP.get(pos_w, pos_w) == POS_GROUP.get(pos_r, pos_r) else None


def tokenize(text):
    out = []
    for w in WORD_RE.findall(text.lower().replace("ё", "е")):
        if w in NEG_WORDS:
            out.append(w)
            continue
        p = _parse(w)
        if p.tag.POS in DROP_POS:
            continue
        rest = split_ne(w)
        if rest:
            out.extend(["не", _parse(rest).normal_form])
        else:
            out.append(p.normal_form)
    return out

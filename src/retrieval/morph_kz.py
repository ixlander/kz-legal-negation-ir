"""Kazakh negative-affix splitter (regex, no full morphology).

A verb form is split as stem + NEG affix (-ма/-ме/-ба/-бе/-па/-пе) + ending only if
  * the affix is followed by a known verbal ending (so «ма» inside ordinary stems is skipped),
  * the affix agrees with the stem: vowel harmony (back -> ма/ба/па, front -> ме/бе/пе)
    and consonant assimilation (vowel/р/й/у/л -> м, м/н/ң/з/ж -> б, voiceless -> п).
Stand-alone question particles (ма/ме/ба/бе/па/пе as a separate word) are never touched.

Precision must be checked on >=100 words together with the owner (CLAUDE.md, 4).
"""
import re

BACK = set("аоұыя")
FRONT = set("әеөүіэ")
VOWELS = BACK | FRONT | set("иуюё")
VOICELESS = set("кқпстфхцчшщ")

# Endings that may follow the negative affix (verb paradigm, participles, verbal noun).
# Order matters: longer first.
ENDINGS = [
    "йтын", "йтін", "йынша", "йінше", "ймын", "ймін", "йсың", "йсің", "йсыз", "йсіз",
    "ймыз", "йміз", "йды", "йді", "й",
    "ған", "ген", "ды", "ді", "ңыз", "ңіз", "у",
]
_END_RE = "|".join(ENDINGS)
_L = "[а-яәіңғүұқөһё]"
# stem (>=2 letters) + affix + ending + optional further suffixes (case, possessive, plural ...);
# the conditional -са/-се only word-final (+ person), otherwise «орынбасары» would match.
_RE = re.compile(rf"^(?P<stem>{_L}{{2,}}?)(?P<affix>[мбп][ае])"
                 rf"(?P<end>(?:{_END_RE}){_L}*|(?:са|се)(?:м|ң|қ|к|ңыз|ңіз)?)$")

# Words that match the pattern but are not negative forms.
STOP_WORDS = {"немесе"}          # 'or'
STOP_STEMS = {"қа"}              # қамау 'arrest', қамауда ...


def _last_vowel(s):
    for ch in reversed(s):
        if ch in VOWELS:
            return ch
    return None


def _expected_onset(stem):
    last = stem[-1]
    if last in VOWELS or last in "рйул":
        return "м"
    if last in "мнңзж":
        return "б"
    if last in VOICELESS:
        return "п"
    return None


def split_neg(word):
    """Return (stem, affix, ending) if `word` looks like a negative verb form, else None."""
    w = word.lower()
    if w in STOP_WORDS:
        return None
    for m in _RE.finditer(w):
        if m.group("stem") in STOP_STEMS:
            continue
        stem, affix, end = m.group("stem"), m.group("affix"), m.group("end")
        v = _last_vowel(stem)
        if v is None:
            continue
        harmony_ok = (affix[1] == "а" and v in BACK | {"у", "и"}) or (affix[1] == "е" and v in FRONT | {"у", "и"})
        if not harmony_ok or _expected_onset(stem) != affix[0]:
            continue
        return stem, affix, end
    return None


def tokenize(text):
    """Lowercase word tokens; negative forms become stem + 'NEG' (+ ending kept on the stem token)."""
    out = []
    for w in re.findall(r"[а-яәіңғүұқөһёa-z0-9]+", text.lower()):
        s = split_neg(w)
        if s:
            out.extend([s[0], "NEG"])
        else:
            out.append(w)
    return out


# --- BM25 analyzer (B2): kazsearch-py stems + explicit NEG -------------------------------
# kazsearch (github.com/iDynbek/kazsearch-py, LGPL-3.0) is installed from a local clone; its
# lexicon lives in the clone's data/ folder (KAZSEARCH_DATA, default ../nlp_tools/kazsearch-py/data).
# Out of the box it strips the negation (төлемейді -> төле) and is not consistent on negative
# forms (аспауға -> аспа, асады -> аса), so for a negative form we stem the verb stem found by
# split_neg and add NEG; other words go through kazsearch unchanged.
import os
from functools import lru_cache
from pathlib import Path

KZ_NEG_PARTICLES = {"емес", "жоқ"}
_KS = {}


def _kazsearch():
    if not _KS:
        from kazsearch import Lexicon, StemConfig, stem
        data = Path(os.environ.get("KAZSEARCH_DATA",
                                   Path(__file__).resolve().parents[3] / "nlp_tools/kazsearch-py/data"))
        _KS["cfg"] = StemConfig(lexicon=Lexicon.load(str(data / "kaz_stems.dict")))
        _KS["stem"] = stem
        _KS["stop"] = set((data / "kaz_stopwords.stop").read_text(encoding="utf-8").split()) - KZ_NEG_PARTICLES
    return _KS


@lru_cache(maxsize=300_000)
def _bm25_word(w):
    ks = _kazsearch()
    if w in KZ_NEG_PARTICLES:
        return (w,)
    if w in ks["stop"]:
        return ()
    s = split_neg(w)
    if s:
        # stem the positive 3sg present form, so «берілмейді» and «беріледі» share the stem
        return (ks["stem"](positive_form(s[0]), ks["cfg"]), "NEG")
    return (ks["stem"](w, ks["cfg"]),)


def positive_form(stem):
    """Verb stem -> positive 3sg present: ас -> асады, беріл -> беріледі, төле -> төлейді."""
    v = _last_vowel(stem)
    back = v in BACK or v is None
    if stem[-1] in VOWELS:
        return stem + ("йды" if back else "йді")
    return stem + ("ады" if back else "еді")


def tokenize_bm25(text):
    out = []
    for w in re.findall(r"[а-яәіңғүұқөһёa-z0-9]+", text.lower()):
        out.extend(_bm25_word(w))
    return out

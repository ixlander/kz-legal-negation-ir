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
    "ған", "ген", "ды", "ді", "са", "се", "ңыз", "ңіз", "у",
]
_END_RE = "|".join(ENDINGS)
# stem (>=2 letters) + affix + ending + optional further suffixes (case, possessive, plural ...)
_RE = re.compile(rf"^(?P<stem>[а-яәіңғүұқөһё]{{2,}}?)(?P<affix>[мбп][ае])(?P<end>(?:{_END_RE})[а-яәіңғүұқөһё]*)$")


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
    for m in _RE.finditer(w):
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

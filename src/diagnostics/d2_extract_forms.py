"""Collect Kazakh negative verb forms from the Kz corpus for D2.

Every form is split by the regex splitter (src/retrieval/morph_kz.py) and gets
status NEEDS_KZ_CHECK; the owner sets status to `ok` / `wrong_split` / `not_neg`.
Rows that are already checked in an existing file are kept as they are.

Usage:
    python -m src.diagnostics.d2_extract_forms
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.retrieval.morph_kz import split_neg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/diagnostics/d2_kz_neg_forms.tsv"
FIELDS = ["form", "stem", "affix", "ending", "freq", "source", "status", "note"]

# Forms that are morphologically negative but lexicalized; flagged for the owner.
LEXICALIZED = ("қоспағанда", "келіспеуші", "қолсұғылмаушы", "сұғылмау")


def main():
    cnt = Counter()
    for path in sorted((ROOT / "data/corpus").glob("*_kz.jsonl")):
        for line in open(path, encoding="utf-8"):
            for w in re.findall(r"[а-яәіңғүұқөһё]+", json.loads(line)["text"].lower()):
                cnt[w] += 1

    old = {}
    if OUT.exists():
        old = {r["form"]: r for r in csv.DictReader(open(OUT, encoding="utf-8"), delimiter="\t")}

    rows = []
    for w, c in cnt.most_common():
        s = split_neg(w)
        if not s:
            continue
        if w in old and old[w]["status"] != "NEEDS_KZ_CHECK":
            rows.append({**old[w], "freq": c})
            continue
        note = "lexicalized?" if w.startswith(LEXICALIZED) else ""
        rows.append({"form": w, "stem": s[0], "affix": s[1], "ending": s[2], "freq": c,
                     "source": "corpus_labor", "status": "NEEDS_KZ_CHECK", "note": note})

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, FIELDS, delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} forms ({sum(r['freq'] for r in rows)} tokens) -> {OUT.relative_to(ROOT)}")
    print(Counter(r["affix"] for r in rows))


if __name__ == "__main__":
    main()

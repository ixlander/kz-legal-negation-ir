"""Owner review of Kazakh drafts.

export: write data/diagnostics/review_d1_kz.tsv (one row per Kz triple, Ru original alongside)
        and data/corpus/labor_alignment_check.tsv (10 random aligned Ru/Kz chunks, seed 42).
apply:  read review_d1_kz.tsv back. For every row the owner marked `ok` (or `fixed` after
        editing the paraphrase/negation/min_paraphrase cells) the triple gets status `ok`;
        `drop` removes it from the metrics (status `dropped`).

Usage:
    python -m src.diagnostics.kz_review export
    python -m src.diagnostics.kz_review apply
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLES = ROOT / "data/diagnostics/d1_triples.jsonl"
REVIEW = ROOT / "data/diagnostics/review_d1_kz.tsv"
ALIGN = ROOT / "data/corpus/labor_alignment_check.tsv"
EDITABLE = ["paraphrase", "negation", "min_paraphrase"]


def load():
    return [json.loads(l) for l in open(TRIPLES, encoding="utf-8") if l.strip()]


def export():
    rows = load()
    ru = {r["tid"]: r for r in rows if r["lang"] == "ru"}
    with open(REVIEW, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tid", "status", "comment", "neg_mechanism", "ru_sentence", "ru_negation",
                    "kz_sentence", *EDITABLE])
        for r in rows:
            if r["lang"] != "kz":
                continue
            p = ru[r["pair_tid"]]
            w.writerow([r["tid"], r["status"], "", r["neg_mechanism"], p["sentence"], p["negation"],
                        r["sentence"], *[r[k] for k in EDITABLE]])

    chunks = {l: [json.loads(x) for x in open(ROOT / f"data/corpus/labor_{l}.jsonl", encoding="utf-8")]
              for l in ("ru", "kz")}
    idx = random.Random(42).sample(range(len(chunks["ru"])), 10)
    with open(ALIGN, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chunk_id", "aligned_ok", "comment", "ru_text", "kz_text"])
        flat = lambda s: s.replace("\n", " / ")[:1500]
        for i in idx:
            w.writerow([chunks["ru"][i]["chunk_id"], "", "", flat(chunks["ru"][i]["text"]),
                        flat(chunks["kz"][i]["text"])])
    print(f"-> {REVIEW.relative_to(ROOT)}\n-> {ALIGN.relative_to(ROOT)}")


def apply():
    review = {r["tid"]: r for r in csv.DictReader(open(REVIEW, encoding="utf-8"), delimiter="\t")}
    rows = load()
    n = {"ok": 0, "dropped": 0, "pending": 0}
    for r in rows:
        rv = review.get(r["tid"])
        if not rv:
            continue
        st = rv["status"].strip().lower()
        if st in ("ok", "fixed"):
            for k in EDITABLE:
                r[k] = rv[k].strip()
            r["status"] = "ok"
            n["ok"] += 1
        elif st == "drop":
            r["status"] = "dropped"
            n["dropped"] += 1
        else:
            n["pending"] += 1
        if rv.get("comment"):
            r["review_comment"] = rv["comment"]
    with open(TRIPLES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(n)


if __name__ == "__main__":
    {"export": export, "apply": apply}[sys.argv[1]]()

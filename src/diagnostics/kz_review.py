"""Owner review of Kazakh drafts.

export: write data/diagnostics/review_d1_kz.tsv (one row per Kz triple, Ru original alongside).
apply:  read review_d1_kz.tsv back. For every row the owner marked `ok` (or `fixed` after
        editing the paraphrase/negation/min_paraphrase cells) the triple gets status `ok`;
        `drop` removes it from the metrics (status `dropped`).

Usage:
    python -m src.diagnostics.kz_review export
    python -m src.diagnostics.kz_review apply
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLES = ROOT / "data/diagnostics/d1_triples.jsonl"
REVIEW = ROOT / "data/diagnostics/review_d1_kz.tsv"
EDITABLE = ["paraphrase", "negation", "min_paraphrase"]


def load():
    return [json.loads(l) for l in open(TRIPLES, encoding="utf-8") if l.strip()]


def export():
    rows = load()
    ru = {r["tid"]: r for r in rows if r["lang"] == "ru"}
    with open(REVIEW, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tid", "status", "comment", "agent_doubt", "neg_mechanism", "ru_sentence", "ru_negation",
                    "kz_sentence", *EDITABLE])
        for r in rows:
            if r["lang"] != "kz":
                continue
            p = ru[r["pair_tid"]]
            w.writerow([r["tid"], r["status"], "", r.get("agent_doubt", ""), r["neg_mechanism"], p["sentence"],
                        p["negation"], r["sentence"], *[r[k] for k in EDITABLE]])
    print(f"-> {REVIEW.relative_to(ROOT)}")


def apply():
    rows_rv = [{k.strip(): (v or "").strip() for k, v in r.items()}
               for r in csv.DictReader(open(REVIEW, encoding="utf-8"), delimiter="\t")]
    review = {r["tid"]: r for r in rows_rv}
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


def apply_d2():
    """data/diagnostics/d2_owner_check.tsv: decision = ok | not_neg | wrong_split -> status in the forms file."""
    forms_path = ROOT / "data/diagnostics/d2_kz_neg_forms.tsv"
    dec = {r["form"]: r for r in csv.DictReader(open(ROOT / "data/diagnostics/d2_owner_check.tsv", encoding="utf-8"),
                                                delimiter="\t")}
    rows = list(csv.DictReader(open(forms_path, encoding="utf-8"), delimiter="\t"))
    n = {"ok": 0, "not_neg": 0, "wrong_split": 0, "pending": 0}
    for r in rows:
        d = dec.get(r["form"])
        if not d:
            continue
        st = d["decision"].strip().lower()
        if st in ("ok", "not_neg", "wrong_split"):
            r["status"] = st
            if d.get("comment"):
                r["note"] = (r["note"] + "; " if r["note"] else "") + d["comment"].strip()
            n[st] += 1
        else:
            n["pending"] += 1
    with open(forms_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(n)


if __name__ == "__main__":
    {"export": export, "apply": apply, "apply-d2": apply_d2}[sys.argv[1]]()

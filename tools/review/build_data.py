"""Collect everything the owner has to review into one JSON for the review page.

Usage:
    python tools/review/build_data.py   -> tools/review/review_data.json
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("review_data.json")


def tsv(path):
    return [{k.strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(open(ROOT / path, encoding="utf-8"), delimiter="\t")]


def corpus(code, lang):
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in open(ROOT / f"data/corpus/{code}_{lang}.jsonl",
                                                                   encoding="utf-8")}


def context(form, chunks):
    """One sentence of the Kz corpus that contains the form."""
    rx = re.compile(r"(?<![а-яәіңғүұқөһё])" + re.escape(form) + r"(?![а-яәіңғүұқөһё])", re.I)
    for r in chunks.values():
        for s in re.split(r"(?<=[.;:])\s+|\n", r["text"]):
            if rx.search(s):
                return {"chunk_id": r["chunk_id"], "text": s[:400]}
    return None


def main():
    kz_all = {}
    for p in sorted((ROOT / "data/corpus").glob("*_kz.jsonl")):
        if p.stem.rsplit("_", 1)[0] in ("labor", "tax"):
            kz_all.update(corpus(p.stem.rsplit("_", 1)[0], "kz"))

    d2 = []
    for i, r in enumerate(tsv("data/diagnostics/d2_owner_check.tsv"), 1):
        d2.append({"key": f"d2-{i:03d}", "form": r["form"], "split": r["regex_split"], "freq": int(r["freq"]),
                   "why": r["why"], "kaznlp": r["kaznlp_says"], "apertium": r["apertium_says"],
                   "context": context(r["form"], kz_all)})

    d1kz = []
    for r in tsv("data/diagnostics/review_d1_kz.tsv"):
        d1kz.append({"key": f"d1-{r['tid']}", "tid": r["tid"], "doubt": r["agent_doubt"],
                     "mechanism": r["neg_mechanism"], "ru_sentence": r["ru_sentence"],
                     "ru_negation": r["ru_negation"], "kz_sentence": r["kz_sentence"],
                     "paraphrase": r["paraphrase"], "negation": r["negation"],
                     "min_paraphrase": r["min_paraphrase"]})

    align = []
    for code in ("labor", "tax"):
        for r in tsv(f"data/corpus/{code}_alignment_flagged.tsv"):
            align.append({"key": "al-" + re.sub(r"[^A-Za-z0-9_.~:@+-]", "_", r["chunk_id"]),
                          "chunk_id": r["chunk_id"], "why": r["why"], "ru": r["ru_text"], "kz": r["kz_text"]})

    labor_ru = corpus("labor", "ru")
    qs = []
    for r in map(json.loads, open(ROOT / "data/eval/queries_draft_ru.jsonl", encoding="utf-8")):
        chunk = lambda c: {"id": c, "title": labor_ru[c]["title"], "text": labor_ru[c]["text"][:1800]}
        qs.append({"key": "q-" + r["qid"], "qid": r["qid"], "slice": r["slice"], "question": r["question"],
                   "gold": [chunk(c) for c in r["gold_ids"]], "twin": [chunk(c) for c in r["twin_ids"]],
                   "notes": r["notes"]})

    d1ru = []
    for r in map(json.loads, open(ROOT / "data/diagnostics/d1_triples.jsonl", encoding="utf-8")):
        if r["lang"] == "ru":
            d1ru.append({"key": f"d1-{r['tid']}", "tid": r["tid"], "chunk_id": r["chunk_id"],
                         "sentence": r["sentence"], "paraphrase": r["paraphrase"], "negation": r["negation"],
                         "min_paraphrase": r["min_paraphrase"], "neg_type": r["neg_type"]})

    data = {"d2": d2, "d1kz": d1kz, "align": align, "questions": qs, "d1ru": d1ru}
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print({k: len(v) for k, v in data.items()}, f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

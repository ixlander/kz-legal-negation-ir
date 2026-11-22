"""Error analysis on the test split: questions whose gold is missed in the top 5.

A question is a case when at least one of the main runs misses its gold at Recall@5. Cases are ranked
by how many runs miss them; the top N are reported with, for the run that shows the error, the top-3
retrieved chunks, the rank of the gold and of the twin. Each case gets an automatic structural tag
from the ranking and a manual error category from the notes file.

Automatic tags (first match):
    twin_on_top     a twin is ranked above the gold and inside the top 5
    same_article    the top-1 chunk is another paragraph of the gold article
    same_code       the top-1 chunk is from the gold's code, another article
    other_code      the top-1 chunk is from one of the 20 distractor codes

Usage:
    python -m src.eval.error_analysis                 # writes the case dump and, if notes exist, the report
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["bge-m3/B0", "bge-m3/B2", "bge-m3/B3'-bge-reranker-v2-m3", "e5-base/B0", "e5-base/B2"]
NOTES = ROOT / "data/eval/error_notes.tsv"
N_CASES = 20


def article(chunk_id):
    code, art = chunk_id.split(":")[:2]
    return code, art.split("#")[0]


def texts():
    out = {}
    for lang in ("ru", "kz"):
        for p in sorted((ROOT / "data/corpus").glob(f"*_{lang}.jsonl")):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                out[lang, r["chunk_id"]] = (r["title"], " ".join(r["text"].split()))
    return out


def tag(gold, twin_rank, gold_rank, top1):
    if twin_rank is not None and twin_rank < gold_rank and twin_rank <= 5:
        return "twin_on_top"
    g = {article(c) for c in gold}
    if article(top1) in g:
        return "same_article"
    if article(top1)[0] in {c for c, _ in g}:
        return "same_code"
    return "other_code"


def cases(test, queries, corpus):
    misses = {}
    for lang, L in test["langs"].items():
        for run in RUNS:
            for r in L["per_query"][run]:
                if r["recall@5"] == 0:
                    misses.setdefault(r["qid"], []).append((run, r))
    ranked = sorted(misses.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:N_CASES]
    out = []
    for qid, runs in ranked:
        q = queries[qid]
        run, r = runs[0]
        lang = q["lang"]
        out.append({
            "qid": qid, "lang": lang, "slice": q["slice"], "question": q["question"],
            "gold_ids": q["gold_ids"], "twin_ids": q["twin_ids"],
            "missed_by": [x for x, _ in runs], "shown_run": run,
            "rank_gold": r["rank_gold"], "rank_twin": r["rank_twin"],
            "tag": tag(q["gold_ids"], r["rank_twin"], r["rank_gold"], r["top5"][0]),
            "gold_text": [corpus[lang, c][1][:300] for c in q["gold_ids"]],
            "top3": [{"id": c, "title": corpus[lang, c][0], "text": corpus[lang, c][1][:300]} for c in r["top5"][:3]],
        })
    return out


def notes():
    if not NOTES.exists():
        return {}
    return {r["qid"]: r for r in csv.DictReader(open(NOTES, encoding="utf-8"), delimiter="\t")}


def markdown(rows):
    cat = Counter(r["category"] for r in rows)
    tags = Counter(r["tag"] for r in rows)
    lines = ["# Error analysis (test split)", "",
             f"{len(rows)} questions with the most Recall@5 misses over {len(RUNS)} runs "
             f"({', '.join(RUNS)}).", "", "## Categories (manual)", "",
             "| category | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cat.most_common()] + \
            ["", "## Structural tags (automatic, on the shown run)", "", "| tag | n |", "|---|---|"] + \
            [f"| {k} | {v} |" for k, v in tags.most_common()] + ["", "## Cases", "",
             "| qid | slice | missed by | gold rank | twin rank | top-1 | tag | category | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['qid']} | {r['slice']} | {len(r['missed_by'])} | {r['rank_gold']} | "
                     f"{r['rank_twin'] if r['rank_twin'] is not None else '–'} | {r['top3'][0]['id']} | "
                     f"{r['tag']} | {r['category']} | {r['note']} |")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    test = json.loads((ROOT / "results/final_ablation_test.json").read_text(encoding="utf-8"))
    queries = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/eval/queries.jsonl", encoding="utf-8"))}
    rows = cases(test, queries, texts())
    (ROOT / "results/error_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = notes()
    if not n:
        print(f"{len(rows)} cases -> results/error_cases.json (no notes yet)")
        return
    missing = [r["qid"] for r in rows if r["qid"] not in n]
    assert not missing, f"no note for {missing}"
    for r in rows:
        r["category"], r["note"] = n[r["qid"]]["category"], n[r["qid"]]["note"]
    out = {"runs": RUNS, "n_cases": len(rows), "categories": dict(Counter(r["category"] for r in rows)),
           "tags": dict(Counter(r["tag"] for r in rows)), "cases": rows}
    (ROOT / "results/error_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "results/error_analysis.md").write_text(markdown(rows), encoding="utf-8")
    print(out["categories"], out["tags"])


if __name__ == "__main__":
    main()

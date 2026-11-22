"""Exact tests and interval estimates for the final test results (reads results only).

    tests      every pre-registered H1-H3 comparison of the final table as an exact two-sided sign test on the
               questions (or question pairs, H3) that changed, with the counts of changes each way; Holm
               correction over the whole family. The Negation Gap has no per-question value and is left out.
    intervals  Wilson 95% intervals for hit@5 (any gold in the top 5) and Twin-above-Gold per cell
    polarity   Twin-above-Gold restricted to the top 5 (twin in the top 5 and above the gold) and the
               polarity miss (twin in the top 5, gold not)
    breakdown  hit@5 by code (Labor / Tax) and, for the Tax pairs, by whether the Russian question uses
               the abbreviations НДС / ИПН / КПН

Usage:
    python -m src.eval.stats_extra
"""
import json
import math
import re
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ["bge-m3", "e5-base"]
LANGS = ["ru", "kz"]
CONTROLS = ["B3'-jina-v2", "B3'-bge-reranker-v2-m3", "B2"]
MAIN_RUNS = ["B0", "B2", "B3'-jina-v2", "B3'-bge-reranker-v2-m3"]
ABBREV = re.compile(r"\b(НДС|ИПН|КПН)\b")


def hit(r):
    return float(r["recall@5"] > 0)


def sign_test(diffs):
    up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    return {"up": up, "down": down, "n": len(diffs), "p": float(p)}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def holm(tests):
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p"]))
        tests[i]["p_holm"] = running
    return tests


def paired(pq, a, b, sl, metric):
    A = {r["qid"]: r for r in pq[a]}
    return [B[metric] - A[B["qid"]][metric] for B in pq[b] if B["slice"] == sl and B[metric] is not None]


def h3_diffs(test, key, run, sl):
    gain = {}
    for lang in LANGS:
        pq = test["langs"][lang]["per_query"]
        base = {r["qid"][3:]: r["recall@5"] for r in pq[f"{key}/B0"] if r["slice"] == sl}
        gain[lang] = {r["qid"][3:]: r["recall@5"] - base[r["qid"][3:]] for r in pq[f"{key}/{run}"] if r["slice"] == sl}
    return [gain["kz"][p] - gain["ru"][p] for p in sorted(gain["ru"])]


def family(test, chosen):
    tests = []
    for key in MODELS:
        b3 = chosen[key]["B3"]
        for lang in LANGS:
            pq = test["langs"][lang]["per_query"]
            for sl in ("negation", "general"):
                tests.append({"test": f"H1 {key} {lang} B2 vs B0 {sl} hit@5",
                              **sign_test(paired(pq, f"{key}/B0", f"{key}/B2", sl, "recall@5"))})
            for ctrl in CONTROLS:
                tests.append({"test": f"H2 {key} {lang} B3 vs {ctrl} negation hit@5",
                              **sign_test(paired(pq, f"{key}/{ctrl}", f"{key}/{b3}", "negation", "recall@5"))})
                tests.append({"test": f"H2 {key} {lang} B3 vs {ctrl} twin-above-gold",
                              **sign_test(paired(pq, f"{key}/{ctrl}", f"{key}/{b3}", "negation", "twin_above_gold"))})
        for run in ("B2", b3):
            for sl in ("negation", "general"):
                tests.append({"test": f"H3 {key} {'B2' if run == 'B2' else 'B3'} kz-ru gain {sl}",
                              **sign_test(h3_diffs(test, key, run, sl))})
    return holm(tests)


def cells(test, chosen):
    out = {}
    for key in MODELS:
        for run in MAIN_RUNS + [chosen[key]["B3"]]:
            for lang in LANGS:
                rows = test["langs"][lang]["per_query"][f"{key}/{run}"]
                cell = {}
                for sl in ("general", "negation"):
                    sub = [r for r in rows if r["slice"] == sl]
                    k = int(sum(hit(r) for r in sub))
                    cell[f"{sl}_hit5"] = {"k": k, "n": len(sub), "rate": k / len(sub), "wilson": wilson(k, len(sub))}
                tw = [r for r in rows if r["twin_above_gold"] is not None]
                tag = int(sum(r["twin_above_gold"] for r in tw))
                tag5 = sum(r["rank_twin"] <= 5 and r["rank_twin"] < r["rank_gold"] for r in tw)
                miss5 = sum(r["rank_twin"] <= 5 and r["rank_gold"] > 5 for r in tw)
                cell["twin_above_gold"] = {"k": tag, "n": len(tw), "wilson": wilson(tag, len(tw))}
                cell["twin_above_gold@5"] = {"k": tag5, "n": len(tw), "wilson": wilson(tag5, len(tw))}
                cell["polarity_miss@5"] = {"k": miss5, "n": len(tw)}
                out[f"{key} | {lang} | {run}"] = cell
    return out


def breakdown(test, queries):
    abbrev = {q["pair_id"] for q in queries if q["lang"] == "ru" and ABBREV.search(q["question"])}
    groups = {"labor": lambda p: p.startswith("lab"),
              "tax, Ru abbreviated": lambda p: p.startswith("tax") and p in abbrev,
              "tax, other": lambda p: p.startswith("tax") and p not in abbrev}
    out = {}
    for key in MODELS:
        for run in MAIN_RUNS:
            for lang in LANGS:
                rows = test["langs"][lang]["per_query"][f"{key}/{run}"]
                for g, f in groups.items():
                    sub = [r for r in rows if f(r["qid"][3:])]
                    k = int(sum(hit(r) for r in sub))
                    out[f"{key} | {run} | {lang} | {g}"] = {"k": k, "n": len(sub), "rate": round(k / len(sub), 3)}
    for lang in LANGS:
        for g, f in groups.items():
            rows = test["langs"][lang]["per_query"]["bm25-morph"]
            sub = [r for r in rows if f(r["qid"][3:])]
            k = int(sum(hit(r) for r in sub))
            out[f"bm25-morph | {lang} | {g}"] = {"k": k, "n": len(sub), "rate": round(k / len(sub), 3)}
    return out


def markdown(fam, cl, br):
    lines = ["# Exact tests and intervals (test split)", "",
             "## Pre-registered comparisons, exact sign test, Holm over all " + str(len(fam)), "",
             "| comparison | better / worse / n | p | p (Holm) |", "|---|---|---|---|"]
    for t in sorted(fam, key=lambda t: t["p"]):
        if t["up"] + t["down"]:
            lines.append(f"| {t['test']} | {t['up']} / {t['down']} / {t['n']} | {t['p']:.3f} | {t['p_holm']:.2f} |")
    lines += ["", "Comparisons with no changed question are omitted.", "",
              "## Twin-above-Gold, full ranking vs top 5 (14 questions with a twin)", "",
              "| cell | TaG k/n [Wilson] | TaG@5 | polarity miss@5 |", "|---|---|---|---|"]
    for name, c in cl.items():
        t, t5, m5 = c["twin_above_gold"], c["twin_above_gold@5"], c["polarity_miss@5"]
        lines.append(f"| {name.replace(' | ', ' · ')} | {t['k']}/{t['n']} {t['wilson']} | {t5['k']}/{t5['n']} | {m5['k']}/{m5['n']} |")
    lines += ["", "## hit@5 by group", "", "| cell | k/n | rate |", "|---|---|---|"]
    lines += [f"| {k.replace(' | ', ' · ')} | {v['k']}/{v['n']} | {v['rate']:.2f} |" for k, v in br.items()]
    return "\n".join(lines) + "\n"


def main():
    test = json.loads((RES / "final_ablation_test.json").read_text(encoding="utf-8"))
    chosen = json.loads((RES / "final_table.json").read_text(encoding="utf-8"))["selected_on_dev"]
    queries = [json.loads(l) for l in open(ROOT / "data/eval/queries.jsonl", encoding="utf-8")]
    fam, cl, br = family(test, chosen), cells(test, chosen), breakdown(test, queries)
    out = {"family_size": len(fam), "tests": fam, "cells": cl, "breakdown": br}
    (RES / "stats_extra.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "stats_extra.md").write_text(markdown(fam, cl, br), encoding="utf-8")
    sig = [t for t in fam if t["p_holm"] < 0.05]
    print(f"{len(fam)} tests, smallest p {min(t['p'] for t in fam):.3f}, significant after Holm: {len(sig)}")


if __name__ == "__main__":
    main()

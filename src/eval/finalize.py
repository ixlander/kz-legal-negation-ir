"""From reviewed annotations to the final numbers, in one run.

Steps (each one is a module of this repo, run as a subprocess and stopped on the first failure):
    1. apply the annotators' answers (finalized questions with a dev/test split, D1/D2 statuses, NLI labels)
    2. D1 on e5-base and on the embedder pool, Ru and Kz (checked triples only)
    3. D2 on XLM-R and on the tokenizer pool (checked forms only)
    4. retrieval ladder on the dev split and on the test split, B3-rule over an alpha grid,
       B3 with the main NLI model in every scoring / fusion
    5. dev selection: one alpha and one B3 variant per dense model, by mean Recall@5 over both slices and
       both languages (ties: lower Twin-above-Gold, then the simpler variant); test is never looked at
    6. final table on test + hypothesis tests:
       H1  B2 vs B0: Negation Gap and general Recall@5
       H2  B3 vs B3′ (each reranker) and vs B3-rule: negation Recall@5 and Twin-above-Gold
       H3  gain of B2 and of B3 over B0 on Kz minus the same gain on Ru, paired by question pair
    7. NLI models on the validation sets and the H5 / H6 comparisons

Usage:
    python -m src.eval.finalize --answers data/review/answers
    python -m src.eval.finalize --skip apply d1 d2        # rerun retrieval and NLI only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.eval.bootstrap import N, SEED, paired_diff
from src.eval.metrics import summary
from src.rerank import cross_encoder, nli_rerank

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02]
NLI_MAIN = "checkpoints/nli-a"
NLI_ALL = ["MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"] + [f"checkpoints/nli-{v}" for v in "abcd"]
DENSE = ["bge-m3", "e5-base"]
SIMPLICITY = ["rrf"] + [f"lin-{lam:g}" for lam in nli_rerank.LAMBDAS] + ["pure"]


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)


def ablation(split):
    out = RES / f"final_ablation_{split}.json"
    run("src.eval.run_ablation", "--queries", "final", "--split", split, "--alphas", *map(str, ALPHAS),
        "--nli", NLI_MAIN, "--out", str(out))
    return json.loads(out.read_text(encoding="utf-8"))


def dev_score(res, run_name):
    """Mean Recall@5 over slices and languages; Twin-above-Gold (lower is better) as the tie-break."""
    r5, tag = [], []
    for L in res["langs"].values():
        s = L["summary"][run_name]
        r5 += [s["general"]["recall@5"], s["negation"]["recall@5"]]
        if s["twin_above_gold"]["value"] is not None:
            tag.append(s["twin_above_gold"]["value"])
    return np.mean(r5), -np.mean(tag) if tag else 0.0


def select(dev):
    nli = Path(NLI_MAIN).name
    chosen = {}
    for key in DENSE:
        alpha = max(ALPHAS, key=lambda a: (*dev_score(dev, f"{key}/B3-rule-{a:g}"), -a))
        b3 = max(nli_rerank.run_names(), key=lambda r: (*dev_score(dev, f"{key}/B3-{nli}-{r}"),
                                                        -SIMPLICITY.index(r.split("-", 1)[1])))
        chosen[key] = {"B3-rule": f"B3-rule-{alpha:g}", "B3": f"B3-{nli}-{b3}", "alpha": alpha, "b3_variant": b3}
    return chosen


def ladder(key, chosen):
    rows = ["B0", "B1", "B2"] + [f"B3'-{rk}" for rk in cross_encoder.RERANKERS]
    return rows + [chosen[key]["B3-rule"], chosen[key]["B3"]]


def h3(test, key, run_name):
    """Kz gain minus Ru gain of `run_name` over B0, Recall@5 per slice, bootstrap over question pairs."""
    out = {}
    for sl in ("general", "negation"):
        gains = {}
        for lang in ("ru", "kz"):
            pq = test["langs"][lang]["per_query"]
            base = {r["qid"][3:]: r["recall@5"] for r in pq[f"{key}/B0"] if r["slice"] == sl}
            new = {r["qid"][3:]: r["recall@5"] for r in pq[f"{key}/{run_name}"] if r["slice"] == sl}
            gains[lang] = {p: new[p] - base[p] for p in base}
        pairs = sorted(set(gains["ru"]) & set(gains["kz"]))
        d = np.array([gains["kz"][p] - gains["ru"][p] for p in pairs])
        idx = np.random.default_rng(SEED).integers(0, len(d), size=(N, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        out[sl] = {"kz_minus_ru_gain": float(d.mean()), "ci95": [float(lo), float(hi)],
                   "excludes_0": bool(lo > 0 or hi < 0), "n_pairs": len(d)}
    return out


def table(test, chosen):
    tab, tests = {}, {}
    for key in DENSE:
        for lang, L in test["langs"].items():
            for name in ladder(key, chosen):
                s = summary(L["per_query"][f"{key}/{name}"])
                tab[f"{key} | {lang} | {name}"] = {
                    "general_r5": s["general"]["recall@5"], "negation_r5": s["negation"]["recall@5"],
                    "negation_gap": s["negation_gap"], "twin_above_gold": s["twin_above_gold"]["value"],
                    "mrr10": s["all"]["mrr@10"], "general_r10": s["general"]["recall@10"],
                    "negation_r10": s["negation"]["recall@10"]}
            pq = L["per_query"]
            b3 = chosen[key]["B3"]
            tests[f"{key} | {lang} | H1 B2 vs B0"] = {
                "/".join(p): paired_diff(pq[f"{key}/B0"], pq[f"{key}/B2"], p)
                for p in [("negation_gap",), ("general", "recall@5"), ("negation", "recall@5")]}
            for ctrl in [f"B3'-{rk}" for rk in cross_encoder.RERANKERS] + [chosen[key]["B3-rule"]]:
                tests[f"{key} | {lang} | H2 B3 vs {ctrl}"] = {
                    "/".join(p): paired_diff(pq[f"{key}/{ctrl}"], pq[f"{key}/{b3}"], p)
                    for p in [("negation", "recall@5"), ("twin_above_gold", "value")]}
        tests[f"{key} | H3 B2"] = h3(test, key, "B2")
        tests[f"{key} | H3 B3"] = h3(test, key, chosen[key]["B3"])
    return tab, tests


def markdown(tab, chosen):
    lines = []
    for key in DENSE:
        lines += [f"\n### {key} (B3-rule α={chosen[key]['alpha']:g}, B3 = {chosen[key]['b3_variant']})\n",
                  "| run | Ru gen R@5 | Ru neg R@5 | Ru gap | Ru TaG | Kz gen R@5 | Kz neg R@5 | Kz gap | Kz TaG |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in ladder(key, chosen):
            cells = []
            for lang in ("ru", "kz"):
                t = tab[f"{key} | {lang} | {name}"]
                tag = "–" if t["twin_above_gold"] is None else f"{t['twin_above_gold']:.2f}"
                cells += [f"{t['general_r5']:.2f}", f"{t['negation_r5']:.2f}", f"{t['negation_gap']:+.2f}", tag]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/review/answers")
    ap.add_argument("--skip", nargs="*", default=[], choices=["apply", "d1", "d2", "retrieval", "nli"])
    a = ap.parse_args()

    if "apply" not in a.skip:
        run("src.data.apply_review", "--answers", a.answers)
    if "d1" not in a.skip:
        for lang in ("ru", "kz"):
            run("src.diagnostics.d1_margin", "--lang", lang, "--tag", "final")
            run("src.diagnostics.d1_pool", "--lang", lang)
    if "d2" not in a.skip:
        run("src.diagnostics.d2_affix", "--tag", "final")
        run("src.diagnostics.d2_pool")
    if "retrieval" not in a.skip:
        dev, test = ablation("dev"), ablation("test")
        chosen = select(dev)
        tab, tests = table(test, chosen)
        out = {"selected_on_dev": chosen, "alpha_grid": ALPHAS, "nli_model": NLI_MAIN, "test_table": tab,
               "hypothesis_tests": tests}
        (RES / "final_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        (RES / "final_table.md").write_text("# Final retrieval table (test split)\n" + markdown(tab, chosen),
                                            encoding="utf-8")
        print(markdown(tab, chosen))
    if "nli" not in a.skip:
        for m in NLI_ALL:
            run("src.rerank.nli_eval", "--model", m, "--tag", f"final-{Path(m).name}")
        run("src.rerank.nli_compare")


if __name__ == "__main__":
    main()

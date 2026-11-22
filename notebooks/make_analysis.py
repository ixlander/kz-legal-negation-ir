"""Generate notebooks/analysis.ipynb: tables and figures of the final results.

The notebook only reads results/ and writes figures to results/figures/; nothing is recomputed.

Usage:
    python notebooks/make_analysis.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

CELLS = [
    ("md", """# Negation-aware retrieval, Ru/Kz legal texts — final analysis

Reads only `results/*.json`; figures go to `results/figures/`. Test split of the reviewed question set v1
(49 pairs: dev 10, test 39 = 20 general + 19 negation per language). Seed 42, paired bootstrap 1000,
95% percentile CI."""),
    ("code", """import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))

# light chart palette: Ru = slot 1, Kz = slot 2; text in neutral ink
RU, KZ, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True,
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
pd.set_option("display.precision", 2)"""),

    ("md", "## 1. Retrieval ladder on test (Recall@5, Negation Gap, Twin-above-Gold)"),
    ("code", """final = load("final_table.json")
rows = []
for k, v in final["test_table"].items():
    model, lang, run = [x.strip() for x in k.split("|")]
    rows.append({"model": model, "lang": lang, "run": run, **v})
tab = pd.DataFrame(rows)
print("selected on dev:", {m: (c["alpha"], c["b3_variant"]) for m, c in final["selected_on_dev"].items()})
tab.pivot_table(index=["model", "run"], columns="lang",
                values=["general_r5", "negation_r5", "twin_above_gold"], sort=False)"""),
    ("code", """LADDER = ["B0", "B1", "B2", "B3'-jina-v2", "B3'-bge-reranker-v2-m3"]
LABEL = {"B0": "B0\\ndense", "B1": "B1\\n+BM25", "B2": "B2\\n+morph", "B3'-jina-v2": "B3′\\njina",
         "B3'-bge-reranker-v2-m3": "B3′\\nbge-rr", "B3": "B3\\nNLI"}

def ladder_runs(model):
    return LADDER + [final["selected_on_dev"][model]["B3"]]

def bars(metric, ylabel, fname, ylim):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, model in zip(axes, ["bge-m3", "e5-base"]):
        runs = ladder_runs(model)
        x = range(len(runs))
        for off, lang, color in ((-0.2, "ru", RU), (0.2, "kz", KZ)):
            vals = [final["test_table"][f"{model} | {lang} | {r}"][metric] for r in runs]
            ax.bar([i + off for i in x], vals, width=0.38, color=color, label={"ru": "Ru", "kz": "Kz"}[lang],
                   edgecolor="white", linewidth=1)
        ax.set_xticks(list(x), [LABEL.get(r, LABEL["B3"]) for r in runs], fontsize=9)
        ax.set_title(model, color=INK, loc="left")
        ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2)
    fig.savefig(FIG / fname)
    plt.show()

bars("negation_r5", "Recall@5, negation questions", "ladder_negation_r5.png", (0, 1.05))
bars("twin_above_gold", "Twin-above-Gold (lower is better)", "ladder_twin_above_gold.png", (0, 0.5))"""),
    ("code", """tests = final["hypothesis_tests"]
out = []
for k, v in tests.items():
    for metric, r in v.items():
        d = r.get("diff", r.get("kz_minus_ru_gain"))
        out.append({"test": k, "metric": metric, "diff": d, "lo": r["ci95"][0], "hi": r["ci95"][1],
                    "CI excludes 0": r["excludes_0"]})
ht = pd.DataFrame(out)
ht[ht["CI excludes 0"]]   # the only differences that count as found"""),
    ("code", "ht"),

    ("md", "## 2. Controls (test): negation tokens in BM25, fusion weight, zero-shot NLI"),
    ("code", """ctl = load("controls_test.json")
names = ["B0", "B2-noneg", "B2", "B2-w0.25", "B2-w0.5", "B2-w1", "B2-w2"]
rows = []
for model in ("bge-m3", "e5-base"):
    for n in names:
        for lang in ("ru", "kz"):
            s = ctl["langs"][lang]["summary"][f"{model}/{n}"]
            rows.append({"model": model, "run": n, "lang": lang, "gen R@5": s["general"]["recall@5"],
                         "neg R@5": s["negation"]["recall@5"], "TaG": s["twin_above_gold"]["value"]})
pd.DataFrame(rows).pivot_table(index=["model", "run"], columns="lang", values=["gen R@5", "neg R@5", "TaG"], sort=False)"""),
    ("code", """rows = []
for model in ("bge-m3", "e5-base"):
    for nli in ("mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "nli-a"):
        for sfx in ("", "-decl"):
            prefix = f"{model}/B3-{nli}{sfx}-"
            for lang in ("ru", "kz"):
                for k, s in ctl["langs"][lang]["summary"].items():
                    if k.startswith(prefix):
                        rows.append({"model": model, "nli": ("zero-shot" if nli != "nli-a" else "fine-tuned a") + sfx,
                                     "lang": lang, "R@5": (s["general"]["recall@5"] + s["negation"]["recall@5"]) / 2,
                                     "TaG": s["twin_above_gold"]["value"]})
nli_runs = pd.DataFrame(rows)
nli_runs.groupby(["model", "nli"])[["R@5", "TaG"]].mean()   # mean over the 10 scoring/fusion variants and 2 langs"""),

    ("md", "## 3. D1: Negation Margin with the size-matched control"),
    ("code", """rows = []
for lang in ("ru", "kz"):
    pool = load(f"d1_pool_{lang}.json")["models"]
    for model, v in pool.items():
        rows.append({"model": model, "lang": lang,
                     "margin (vs paraphrase)": v["margin"]["mean"], "lo": v["margin"]["mean_ci95"][0],
                     "hi": v["margin"]["mean_ci95"][1], "share < 0": v["margin"]["frac_negative"],
                     "margin (vs one-word synonym)": v["margin_min"]["mean"],
                     "control share < 0": v["margin_min"]["frac_negative"]})
d1 = pd.DataFrame(rows)
d1"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
models = list(dict.fromkeys(d1["model"]))
for ax, (col, title) in zip(axes, [("margin (vs paraphrase)", "cos(s, paraphrase) − cos(s, negation)"),
                                   ("margin (vs one-word synonym)", "cos(s, one-word synonym) − cos(s, negation)")]):
    for off, lang, color in ((-0.2, "ru", RU), (0.2, "kz", KZ)):
        sub = d1[d1.lang == lang].set_index("model").loc[models]
        ax.bar([i + off for i in range(len(models))], sub[col], width=0.38, color=color, label=lang.capitalize(),
               edgecolor="white", linewidth=1)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=9)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
axes[0].set_ylabel("mean margin (below 0: negation is closer)")
axes[0].legend(frameon=False)
fig.savefig(FIG / "d1_margin_with_control.png")
plt.show()"""),

    ("md", "## 4. D2: Affix Isolation Rate (171 checked Kazakh negative forms)"),
    ("code", """d2 = load("d2_pool.json")["tokenizers"]
pd.DataFrame([{"tokenizer": k, "isolation": v["kz"]["isolation_rate"], "any boundary": v["kz"]["any_boundary_rate"],
               "freq-weighted isolation": v["kz"]["isolation_rate_freq_weighted"]} for k, v in d2.items()]
             ).sort_values("isolation", ascending=False)"""),

    ("md", "## 5. NLI models: validation sets, H5 and H6"),
    ("code", """files = {"start (zero-shot)": "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "a": "nli-a", "b": "nli-b",
         "c": "nli-c", "d": "nli-d"}
rows = []
for label, tag in files.items():
    s = load(f"nli_eval_final-{tag}_2026-11-16.json")["sets"]
    rows.append({"model": label, "Kardeş kk": s["kardes-kk-report"]["accuracy"], "LiDiRus neg": s["lidirus-neg"]["accuracy"],
                 "manual ru": s["manual-ru"]["accuracy"], "manual kz": s["manual-kz"]["accuracy"],
                 "synthetic ru": s["synthetic-ru-test"]["accuracy"],
                 "synthetic ru range err": s["synthetic-ru-test"].get("range_negation_error")})
pd.DataFrame(rows).set_index("model")"""),
    ("code", """h = load("nli_h5_h6_2026-11-16.json")["comparisons"]
pd.DataFrame([{"hyp": c["hypothesis"], "pair": f"{c['b']} vs {c['a']}", "set": c["set"], "metric": c["metric"],
               "diff": c["diff"], "lo": c["ci95"][0], "hi": c["ci95"][1], "n": c["n"], "CI excludes 0": c["excludes_0"]}
              for c in h])"""),

    ("md", "## 6. External checks: the pipeline reproduces published numbers"),
    ("code", """rows = []
for name, file in (("RuLawRetrieval tk_hard", "external_rulaw_bge-m3+e5-base_2026-11-14.json"),
                   ("KazQAD (subsample)", "external_kazqad_bge-m3+e5-base_2026-11-14.json")):
    for model, v in load(file)["results"].items():
        rows.append({"benchmark": name, "model": model, "nDCG@10": v["ndcg@10"]})
display(pd.DataFrame(rows).pivot(index="model", columns="benchmark", values="nDCG@10"))
pd.Series(load("nevir_check_2026-11-15.json")["results"], name="NevIR pairwise acc (random 0.25)")"""),

    ("md", "## 7. Error analysis (20 test misses)"),
    ("code", """err = load("error_analysis.json")
cats = pd.Series(err["categories"]).sort_values()
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(cats.index, cats.values, color=RU, height=0.6)
ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
for i, v in enumerate(cats.values):
    ax.text(v + 0.1, i, str(v), va="center", color=INK, fontsize=10)
ax.set_xlabel("questions")
fig.savefig(FIG / "error_categories.png")
plt.show()
pd.DataFrame(err["cases"])[["qid", "slice", "rank_gold", "rank_twin", "tag", "category", "note"]]"""),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / "analysis.ipynb")
    print("-> notebooks/analysis.ipynb")


if __name__ == "__main__":
    main()

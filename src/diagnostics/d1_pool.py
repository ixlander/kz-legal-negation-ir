"""D1 Negation Margin on the whole embedding pool (src/models.py EMBEDDERS).

Same triples and definitions as d1_margin.py; all three sentences are encoded as documents.
Writes one JSON with per-model summaries (and per-triple margins) per language.

Usage:
    python -m src.diagnostics.d1_pool --lang ru
    python -m src.diagnostics.d1_pool --lang kz --include-unchecked
"""
import argparse
import gc
import json
from datetime import date

import numpy as np
import torch

from src.diagnostics.d1_margin import ROOT, SEED, load_triples, summarize
from src.models import EMBEDDERS, encode_docs, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--include-unchecked", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(EMBEDDERS))
    args = ap.parse_args()

    triples = load_triples(ROOT / "data/diagnostics/d1_triples.jsonl", args.lang, args.include_unchecked)
    out = {"diagnostic": "D1_negation_margin_pool",
           "config": {"lang": args.lang, "n": len(triples), "include_unchecked": args.include_unchecked,
                      "encoding": "documents (model-specific prefix/prompt, src/models.py)", "seed": SEED,
                      "bootstrap_resamples": 1000, "date": date.today().isoformat()},
           "models": {}}
    for key in args.models:
        try:
            m = load(key, device="cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            out["models"][key] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{key:<20} LOAD FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        enc = lambda field: encode_docs(m, key, [t[field] for t in triples])
        S, P, N, M = enc("sentence"), enc("paraphrase"), enc("negation"), enc("min_paraphrase")
        cp, cn, cm = (S * P).sum(1), (S * N).sum(1), (S * M).sum(1)
        margin, margin_min = cp - cn, cm - cn
        out["models"][key] = {
            "id": EMBEDDERS[key]["id"],
            "margin": summarize(margin), "margin_min": summarize(margin_min),
            "dist_mean": {"synonym": float((1 - cm).mean()), "negation": float((1 - cn).mean()),
                          "paraphrase": float((1 - cp).mean())},
            # scale-free version: where does the negation sit between synonym and paraphrase distance
            "negation_position": float(np.median(((1 - cn) - (1 - cm)) / np.maximum((1 - cp) - (1 - cm), 1e-9))),
            "per_triple": {t["tid"]: [float(a), float(b)] for t, a, b in zip(triples, margin, margin_min)},
        }
        s = out["models"][key]
        print(f"{key:<20} margin {s['margin']['mean']:+.4f} [{s['margin']['mean_ci95'][0]:+.4f},"
              f"{s['margin']['mean_ci95'][1]:+.4f}]  <0: {s['margin']['frac_negative']:.0%}   "
              f"control {s['margin_min']['mean']:+.4f} (<0: {s['margin_min']['frac_negative']:.0%})   "
              f"neg-pos {s['negation_position']:.2f}")
        del m
        gc.collect()
        torch.cuda.empty_cache()
    tag = "" if not args.include_unchecked else "_prelim"
    (ROOT / f"results/d1_pool_{args.lang}{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")


if __name__ == "__main__":
    main()

"""D1: paired Ru vs Kz comparison on the same norms (reads results/, computes nothing on models).

delta = margin_kz - margin_ru per parallel pair; paired bootstrap 95% CI.

Usage:
    python -m src.diagnostics.d1_compare --ru results/d1_margin_pilot_ru.json --kz results/d1_margin_prelim_kz.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.diagnostics.d1_margin import ROOT, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru", required=True)
    ap.add_argument("--kz", required=True)
    ap.add_argument("--out", default="results/d1_compare_ru_kz.json")
    args = ap.parse_args()

    ru = {r["tid"][2:]: r for r in json.load(open(ROOT / args.ru, encoding="utf-8"))["per_triple"]}
    kz = {r["tid"][2:]: r for r in json.load(open(ROOT / args.kz, encoding="utf-8"))["per_triple"]}
    ids = sorted(set(ru) & set(kz))
    out = {"n_pairs": len(ids), "ru_file": args.ru, "kz_file": args.kz, "definition": "delta = kz - ru, paired"}
    for key in ("margin", "margin_min"):
        d = np.array([kz[i][key] - ru[i][key] for i in ids])
        out[key] = {"mean_delta": float(d.mean()), "ci95": bootstrap_ci(d),
                    "frac_kz_lower": float((d < 0).mean()),
                    "ru_mean": float(np.mean([ru[i][key] for i in ids])),
                    "kz_mean": float(np.mean([kz[i][key] for i in ids]))}
    Path(ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

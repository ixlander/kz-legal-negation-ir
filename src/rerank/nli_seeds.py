"""Extra training seeds for the NLI variants, run in pieces that fit a time budget.

Each job trains one variant with one seed and evaluates it. Finished jobs are skipped, so the
command can be repeated on different days until everything is done; a job is started only if
its estimated duration still fits into the remaining budget.

Usage:
    python -m src.rerank.nli_seeds --hours 3              # do what fits into 3 hours
    python -m src.rerank.nli_seeds --hours 3 --variants a c b --seeds 1 2
    python -m src.rerank.nli_seeds --report               # mean and spread over the finished seeds
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CKPT, RES = ROOT / "checkpoints", ROOT / "results"
BASE_SEED = 42
DEFAULT_HOURS = {"a": 2.6, "b": 1.5, "c": 2.6, "d": 3.8}   # two epochs on a 6 GB laptop GPU
METRICS = [("kardes-kk-report", "accuracy"), ("manual-ru", "accuracy"), ("manual-kz", "accuracy"),
           ("synthetic-ru-test", "range_negation_error"), ("manual-kz", "range_negation_error")]


def name(variant, seed):
    return f"nli-{variant}" if seed == BASE_SEED else f"nli-{variant}-s{seed}"


def eval_file(variant, seed):
    files = sorted(glob.glob(str(RES / f"nli_eval_final-{name(variant, seed)}_*.json")))
    return Path(files[-1]) if files else None


def finished(variant, seed):
    return (CKPT / name(variant, seed) / "train_log.json").exists() and eval_file(variant, seed) is not None


def estimate_hours(variant):
    """Duration of the last finished run of this variant, else a default."""
    logs = sorted(CKPT.glob(f"nli-{variant}*/train_log.json"), key=lambda p: p.stat().st_mtime)
    if logs:
        seconds = sum(h.get("seconds", 0) for h in json.loads(logs[-1].read_text(encoding="utf-8"))["history"])
        if seconds:
            return seconds / 3600 * 1.1
    return DEFAULT_HOURS[variant]


def run_jobs(variants, seeds, hours):
    start = time.time()
    for seed in seeds:
        for variant in variants:
            if finished(variant, seed):
                print(f"{name(variant, seed)}: done, skipped")
                continue
            need, used = estimate_hours(variant), (time.time() - start) / 3600
            if used + need > hours:
                print(f"{name(variant, seed)}: needs ~{need:.1f} h, {hours - used:.1f} h left -> stop here")
                return
            print(f"{name(variant, seed)}: training (~{need:.1f} h)", flush=True)
            for module, args in (("src.rerank.nli_train", ["--variant", variant, "--seed", str(seed)]),
                                 ("src.rerank.nli_eval", ["--model", str(CKPT / name(variant, seed)),
                                                          "--tag", f"final-{name(variant, seed)}"])):
                subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)
    print("all jobs done")


def report(variants, seeds):
    out = {}
    for variant in variants:
        runs = {s: json.loads(eval_file(variant, s).read_text(encoding="utf-8"))["sets"]
                for s in [BASE_SEED, *seeds] if finished(variant, s) or (s == BASE_SEED and eval_file(variant, s))}
        out[variant] = {"seeds": sorted(runs)}
        for set_name, metric in METRICS:
            values = [r[set_name][metric] for r in runs.values() if metric in r.get(set_name, {})]
            if values:
                out[variant][f"{set_name}/{metric}"] = {"mean": float(np.mean(values)), "min": min(values),
                                                        "max": max(values), "values": values}
                print(f"{variant} {set_name:18s} {metric:21s} mean {np.mean(values):.3f} "
                      f"[{min(values):.3f}, {max(values):.3f}] n={len(values)}")
    (RES / "nli_seeds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("-> results/nli_seeds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["a", "c"], choices=list("abcd"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--hours", type=float, default=3.0, help="time budget of this sitting")
    ap.add_argument("--report", action="store_true", help="summarize finished seeds instead of training")
    a = ap.parse_args()
    if a.report:
        report(a.variants, a.seeds)
    else:
        run_jobs(a.variants, a.seeds, a.hours)


if __name__ == "__main__":
    main()

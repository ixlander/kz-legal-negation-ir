"""Automatic Ru/Kz alignment check of ALL aligned chunks (replaces the manual 10-pair sample, CLAUDE.md 5.1).

Per pair:
  markers  point / sub-point numbering («1.», «2)», «7-1)») must be the same sequence
  numbers  multiset of digit groups (dates, sums, article refs) must match
  length   Kz/Ru character ratio
  xling    e5 cross-lingual retrieval: rank of the true Kz chunk for the Ru chunk among all
           Kz chunks of the code, and vice versa (1 = it is the nearest one)
A pair goes to the owner if markers or numbers differ or the ratio is outside [0.6, 1.6].
A low cross-lingual rank alone (markers and numbers identical) is reported but not sent:
it means a near-duplicate neighbour article, not a misalignment.

Usage:
    python -m src.scrape.check_alignment --code labor
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "intfloat/multilingual-e5-base"
MARKER_RE = re.compile(r"(?:^|\n)\s*(\d+(?:-\d+)?[.)])")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def markers(t):
    return MARKER_RE.findall(t)


def numbers(t):
    t = MARKER_RE.sub("\n", t)
    # thousands separators and spacing differ between the versions: «1000» vs «1 000», «121, 122» vs «121,122»
    t = re.sub(r"(?<=\d)[  ](?=\d)", "", t)
    t = re.sub(r"(?<=\d),\s*(?=\d{3}\b)", " ", t)
    return Counter(NUM_RE.findall(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="labor")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    L = {l: [json.loads(x) for x in open(ROOT / f"data/corpus/{args.code}_{l}.jsonl", encoding="utf-8")]
         for l in ("ru", "kz")}
    assert [r["chunk_id"] for r in L["ru"]] == [r["chunk_id"] for r in L["kz"]]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    enc = lambda rows: m.encode(["passage: " + r["text"] for r in rows], normalize_embeddings=True,
                                batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    R, K = enc(L["ru"]), enc(L["kz"])
    S = R @ K.T
    diag = np.diag(S)
    rank_r2k = (S > diag[:, None]).sum(1) + 1
    rank_k2r = (S.T > diag[:, None]).sum(1) + 1

    per, flagged = [], []
    for i, (ru, kz) in enumerate(zip(L["ru"], L["kz"])):
        mk_ok = markers(ru["text"]) == markers(kz["text"])
        nr, nk = numbers(ru["text"]), numbers(kz["text"])
        num_ok = nr == nk
        ratio = len(kz["text"]) / max(1, len(ru["text"]))
        reasons = []
        if not mk_ok:
            reasons.append("markers differ")
        if not num_ok:
            reasons.append(f"numbers differ: ru-only {dict(nr - nk)} kz-only {dict(nk - nr)}")
        if not 0.6 <= ratio <= 1.6:
            reasons.append(f"length ratio {ratio:.2f}")
        if rank_r2k[i] > args.top or rank_k2r[i] > args.top:
            reasons.append(f"xling rank ru->kz {rank_r2k[i]}, kz->ru {rank_k2r[i]}")
        row = {"chunk_id": ru["chunk_id"], "markers_ok": mk_ok, "numbers_ok": num_ok, "ratio": round(ratio, 3),
               "cos": round(float(diag[i]), 4), "rank_ru2kz": int(rank_r2k[i]), "rank_kz2ru": int(rank_k2r[i]),
               "flag_reasons": reasons}
        per.append(row)
        if any(not r.startswith("xling") for r in reasons):
            flagged.append((ru, kz, reasons))

    n = len(per)
    summary = {
        "code": args.code, "pairs": n, "model": MODEL, "top": args.top,
        "markers_match": sum(p["markers_ok"] for p in per) / n,
        "numbers_match": sum(p["numbers_ok"] for p in per) / n,
        "xling_rank1_both": sum(p["rank_ru2kz"] == 1 and p["rank_kz2ru"] == 1 for p in per) / n,
        "xling_top3_both": sum(p["rank_ru2kz"] <= 3 and p["rank_kz2ru"] <= 3 for p in per) / n,
        "flagged_any": sum(bool(p["flag_reasons"]) for p in per),
        "sent_to_owner": len(flagged),
        "flag_reason_counts": dict(Counter(r.split(":")[0].split(" ru->")[0].rsplit(" ", 1)[0] if r.startswith("length")
                                           else r.split(":")[0].split(" ru->")[0]
                                           for p in per for r in p["flag_reasons"])),
    }
    (ROOT / f"results/alignment_check_{args.code}.json").write_text(
        json.dumps({"summary": summary, "per_pair": per}, ensure_ascii=False, indent=2), encoding="utf-8")

    flat = lambda s: s.replace("\n", " / ")[:1500]
    with open(ROOT / f"data/corpus/{args.code}_alignment_flagged.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["chunk_id", "why", "aligned_ok", "comment", "ru_text", "kz_text"])
        for ru, kz, reasons in flagged:
            w.writerow([ru["chunk_id"], "; ".join(reasons), "", "", flat(ru["text"]), flat(kz["text"])])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for ru, kz, reasons in flagged:
        print(f"  {ru['chunk_id']:<16} {'; '.join(reasons)}")


if __name__ == "__main__":
    main()

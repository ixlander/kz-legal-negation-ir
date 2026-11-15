"""Apply the annotators' answers from the review page to the project data.

Input: the page's `reviews` collection exported as one JSON file per card (doc id = card key), e.g.
D2 form `d2-001`, Kz triple `d1-kz001`, alignment `al-labor_26_1`, question `q-ru_lab_001`,
NLI pair `nli-v001`, Ru triple `d1-ru001`. Every card carries `decision`, optional `edits` and `comment`.

Effects:
    D2 forms        ok -> status ok; wrong_split -> split corrected from «stem + affix + ending», ok;
                    not_neg -> excluded
    D1 triples      ok / fixed -> status ok with the edited texts; drop -> status dropped (Kz and Ru)
    alignment       decisions and comments stored next to the corpus
    questions       finalized set: reviewed pairs only, edits applied, ids validated,
                    5 pairs per slice -> dev, the rest -> test
    NLI validation  the annotator's label for both languages, edited Kz hypothesis, status ok;
                    drop -> status dropped
Unreviewed cards keep their draft status and stay out of every metric.

Usage:
    python -m src.data.apply_review --answers data/review/answers
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
SEED, DEV_PAIRS_PER_SLICE = 42, 5
NLI_LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
FINAL_DECISIONS = {"ok", "fixed"}


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def write_jl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_answers(folder):
    out = {}
    for p in Path(folder).rglob("*.json"):
        if p.name.startswith("."):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("decision"):
            out[unquote(p.stem)] = data   # file name = URL-encoded card key
    return out


def tsv(path):
    return [{k.strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")]


def apply_d2(ans, warn):
    owner = tsv(D / "diagnostics/d2_owner_check.tsv")
    decisions = {r["form"]: ans.get(f"d2-{i:03d}") for i, r in enumerate(owner, 1)}
    path = D / "diagnostics/d2_kz_neg_forms.tsv"
    rows = tsv(path)
    n = Counter()
    for r in rows:
        a = decisions.get(r["form"])
        if not a:
            continue
        if a["decision"] == "ok":
            r["status"] = "ok"
        elif a["decision"] == "not_neg":
            r["status"] = "not_neg"
        elif a["decision"] == "wrong_split":
            parts = [p.strip() for p in (a.get("edits") or {}).get("correct_split", "").split("+")]
            if len(parts) == 3 and "".join(parts) == r["form"]:
                r["stem"], r["affix"], r["ending"], r["status"] = *parts, "ok"
            else:
                r["status"] = "wrong_split"
                warn.append(f"D2 {r['form']}: wrong split without a usable correction ({parts})")
        if a.get("comment"):
            r["note"] = "; ".join(x for x in (r.get("note", ""), a["comment"]) if x)
        n[r["status"]] += 1
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return dict(n)


def apply_d1(ans):
    path = D / "diagnostics/d1_triples.jsonl"
    rows = jl(path)
    n = Counter()
    for r in rows:
        a = ans.get(f"d1-{r['tid']}")
        if not a:
            continue
        if a["decision"] in FINAL_DECISIONS:
            for field, text in (a.get("edits") or {}).items():
                if field in ("paraphrase", "negation", "min_paraphrase") and text.strip():
                    r[field] = text.strip()
            r["status"] = "ok"
        elif a["decision"] == "drop":
            r["status"] = "dropped"
        if a.get("comment"):
            r["review_comment"] = a["comment"]
        n[f"{r['lang']}:{r['status']}"] += 1
    write_jl(path, rows)
    return dict(n)


def apply_alignment(ans):
    out = {k[3:]: {"decision": a["decision"], "comment": a.get("comment", "")}
           for k, a in ans.items() if k.startswith("al-")}
    (D / "corpus/alignment_review.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def card_id(chunk_id):
    """Chunk id as it appears in a review-page card key."""
    return re.sub(r"[^A-Za-z0-9_.~:@+-]", "_", chunk_id)


def corpus_ids():
    ids = set()
    for code in ("labor", "tax"):
        for lang in ("ru", "kz"):
            ids |= {json.loads(l)["chunk_id"] for l in open(D / f"corpus/{code}_{lang}.jsonl", encoding="utf-8")}
    return ids


def apply_questions(ans, alignment, warn):
    ru = jl(D / "eval/queries_draft_ru.jsonl")
    kz = {r["pair_id"]: r for r in jl(D / "eval/queries_draft_kz.jsonl")}
    ids = corpus_ids()
    misaligned = {k for k, v in alignment.items() if v["decision"] == "no"}
    final = []
    for r in ru:
        a = ans.get(f"q-{r['qid']}")
        if not a or a["decision"] not in FINAL_DECISIONS:
            continue
        e = a.get("edits") or {}
        gold = e.get("gold", " ".join(r["gold_ids"])).split()
        twin = e.get("twin", " ".join(r["twin_ids"])).split()
        bad = [c for c in gold + twin if c not in ids]
        if bad or not gold:
            warn.append(f"{r['qid']}: dropped, unknown or empty ids {bad or 'gold'}")
            continue
        for c in {c for c in gold + twin if card_id(c) in misaligned}:
            warn.append(f"{r['qid']}: {c} was marked as Ru/Kz misaligned")
        note = "; ".join(x for x in (r["notes"], a.get("comment", "")) if x)
        common = {"pair_id": r["pair_id"], "slice": r["slice"], "gold_ids": gold, "twin_ids": twin, "notes": note}
        final.append({**common, "qid": r["qid"], "lang": "ru", "question": e.get("question", r["question"]).strip()})
        final.append({**common, "qid": kz[r["pair_id"]]["qid"], "lang": "kz",
                      "question": e.get("question_kz", kz[r["pair_id"]]["question"]).strip()})
    rng = random.Random(SEED)
    dev = set()
    for sl in ("general", "negation"):
        pairs = sorted({q["pair_id"] for q in final if q["slice"] == sl})
        if sl == "negation":   # dev pairs with a twin, so Twin-above-Gold is defined on dev
            with_twin = sorted({q["pair_id"] for q in final if q["slice"] == sl and q["twin_ids"]})
            pairs = with_twin if len(with_twin) >= DEV_PAIRS_PER_SLICE else pairs
        dev |= set(rng.sample(pairs, min(DEV_PAIRS_PER_SLICE, len(pairs))))
    for q in final:
        q["split"] = "dev" if q["pair_id"] in dev else "test"
    write_jl(D / "eval/queries.jsonl", final)
    n_pairs = len(final) // 2
    if n_pairs < 35:
        warn.append(f"only {n_pairs} finalized question pairs (fallback minimum in the plan: 35)")
    return dict(Counter(f"{q['split']}:{q['slice']}" for q in final if q["lang"] == "ru"))


def apply_nli(ans):
    path = D / "nli/val_manual.jsonl"
    rows = jl(path)
    n = Counter()
    for r in rows:
        a = ans.get(f"nli-{r['pair_id']}")
        if not a:
            continue
        if a["decision"] == "drop":
            r["status"] = "dropped"
        elif a["decision"] in NLI_LABELS:
            r["label_draft"] = r.get("label_draft", r["label"])
            r["label"] = NLI_LABELS[a["decision"]]
            r["status"] = "ok"
            hyp = (a.get("edits") or {}).get("hypothesis_kz", "").strip()
            if r["lang"] == "kz" and hyp:
                r["hypothesis"] = hyp
        if a.get("comment"):
            r["review_comment"] = a["comment"]
        n[f"{r['lang']}:{r['status']}"] += 1
    write_jl(path, rows)
    agree = [r["label"] == r["label_draft"] for r in rows if r["lang"] == "ru" and r["status"] == "ok"]
    return {**dict(n), "agreement_with_draft_label": round(sum(agree) / len(agree), 3) if agree else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=D / "review/answers")
    a = ap.parse_args()
    ans = load_answers(a.answers)
    warn = []
    summary = {"cards_answered": len(ans), "by_section": dict(Counter(re.sub(r"-.*", "", k) for k in ans))}
    summary["d2"] = apply_d2(ans, warn)
    summary["d1"] = apply_d1(ans)
    alignment = apply_alignment(ans)
    summary["alignment"] = dict(Counter(v["decision"] for v in alignment.values()))
    summary["questions"] = apply_questions(ans, alignment, warn)
    summary["nli"] = apply_nli(ans)
    summary["warnings"] = warn
    (ROOT / "results/review_applied.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

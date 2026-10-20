"""Cross-check the regex negative-affix splitter against two independent Kazakh analyzers.

Neither tool is vendored; pass local paths.
  * kaznlp (github.com/nlacslab/kaznlp): segmented analyses, negative affix tagged *_ETB
    («бер_R_ET іл_V2 ме_ET_ETB й_T1 ді_P3») -> checks negation AND the affix position.
  * apertium-kaz (nightly deb, kaz.automorf.hfst, read with pyhfst): lemma + tags
    («беріл<v><tv><neg><aor><p3><sg>») -> checks negation; the position only through
    "the regex stem starts with the lemma".

A form is auto-confirmed when
  kaznlp agrees (negative, same position) and apertium does not say "not negative", or
  kaznlp does not know the word and apertium says negative in every analysis with a consistent stem.
Everything else — and every form flagged `lexicalized?` — goes to the owner.

Usage:
    python -m src.diagnostics.d2_crosscheck --kaznlp ../nlp_tools/kaznlp \
        --apertium ../nlp_tools/apertium_kaz/usr/share/apertium/apertium-kaz/kaz.automorf.hfst
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORMS = ROOT / "data/diagnostics/d2_kz_neg_forms.tsv"
OWNER = ROOT / "data/diagnostics/d2_owner_check.tsv"


def kaznlp_neg_spans(analysis):
    """'бер_R_ET іл_V2 ме_ET_ETB й_T1 ді_P3' -> {(start, end)} of ETB morphs."""
    pos, out = 0, set()
    for seg in analysis.split():
        morph, _, tag = seg.partition("_")
        if tag.endswith("ETB") and tag != "R_ETB":
            out.add((pos, pos + len(morph)))
        pos += len(morph)
    return out


def kaznlp_verdict(analyses, covered, a, b):
    if not covered:
        return "not_covered"
    negs = [kaznlp_neg_spans(x) for x in analyses]
    if not any(negs):
        return "not_neg"
    if not all(negs):
        return "ambiguous"
    return "agree_neg" if all((a, b) in n for n in negs) else "split_differs"


def apertium_verdict(analyses, stem, affix):
    """neg / neg_lex / ambiguous / not_neg / not_covered.

    neg_lex: every analysis is either <neg> or a lexicalized entry whose lemma already
    contains stem+affix (реттелмеген<adj>, келме<v>) — the affix is still in the word,
    which is what D2 measures.
    """
    if not analyses or all("<" not in x for x in analyses):
        return "not_covered", False
    lemma = lambda x: re.match(r"[^<+]*", x).group(0)
    neg = ["<neg>" in x for x in analyses]
    lex = [not n and lemma(x).startswith(stem + affix) for x, n in zip(analyses, neg)]
    lemmas = {lemma(x) for x, n in zip(analyses, neg) if n}
    stem_ok = any(stem.startswith(l) and l for l in lemmas) or any(lex)
    if all(neg):
        return "neg", stem_ok
    if all(n or l for n, l in zip(neg, lex)):
        return "neg_lex", stem_ok
    return ("ambiguous" if any(neg) or any(lex) else "not_neg"), stem_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaznlp", required=True, help="path to a local clone of nlacslab/kaznlp")
    ap.add_argument("--apertium", required=True, help="path to kaz.automorf.hfst from the apertium-kaz package")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.kaznlp))
    from kaznlp.morphology.analyzers import AnalyzerDD
    import pyhfst
    kz = AnalyzerDD()
    kz.load_model(os.path.join(args.kaznlp, "kaznlp", "morphology", "mdl"))
    apt = pyhfst.HfstInputStream(args.apertium).read()

    rows = list(csv.DictReader(open(FORMS, encoding="utf-8"), delimiter="\t"))
    per, cnt, cnt_w = [], Counter(), Counter()
    for r in rows:
        a = len(r["stem"])
        b = a + len(r["affix"])
        covered, k_an = kz.analyze(r["form"])
        kv = kaznlp_verdict(k_an, covered, a, b)
        a_an = [x[0] for x in apt.lookup(r["form"])]
        av, stem_ok = apertium_verdict(a_an, r["stem"], r["affix"])
        auto = ((kv == "agree_neg" and av != "not_neg")
                or (kv == "not_covered" and av in ("neg", "neg_lex") and stem_ok))
        lex = "lexicalized" in r.get("note", "")
        final = "auto_ok" if auto and not lex else "owner"
        cnt[final] += 1
        cnt_w[final] += int(r["freq"])
        per.append({"form": r["form"], "regex_split": f"{r['stem']}+{r['affix']}+{r['ending']}", "freq": int(r["freq"]),
                    "kaznlp": kv, "apertium": av, "apertium_stem_ok": stem_ok, "lexicalized_flag": lex,
                    "final": final, "kaznlp_analyses": k_an[:4], "apertium_analyses": a_an[:4]})

    out = {"tools": {"kaznlp": "AnalyzerDD, github.com/nlacslab/kaznlp (HEAD, Oct 2026)",
                     "apertium": "apertium-kaz 0.1.0+g2602~95c6dd0d nightly, kaz.automorf.hfst via pyhfst 1.4.0"},
           "forms_file": str(FORMS.relative_to(ROOT)), "n": len(rows),
           "final": dict(cnt), "final_freq_weighted": dict(cnt_w),
           "kaznlp": dict(Counter(p["kaznlp"] for p in per)),
           "apertium": dict(Counter(p["apertium"] for p in per)), "per_form": per}
    (ROOT / "results/d2_crosscheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(OWNER, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["form", "regex_split", "freq", "why", "kaznlp_says", "apertium_says", "decision", "comment"])
        for p in per:
            if p["final"] != "owner":
                continue
            why = "lexicalized?" if p["lexicalized_flag"] else f"kaznlp={p['kaznlp']}, apertium={p['apertium']}"
            w.writerow([p["form"], p["regex_split"], p["freq"], why, " | ".join(p["kaznlp_analyses"][:2]),
                        " | ".join(p["apertium_analyses"][:2]), "", ""])
    # write the verdict back: still-unchecked rows become ok_auto (both tools) or stay for the owner
    verdict = {p["form"]: p["final"] for p in per}
    for r in rows:
        if r["status"] == "NEEDS_KZ_CHECK" and verdict[r["form"]] == "auto_ok":
            r["status"] = "ok_auto"
    with open(FORMS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({k: out[k] for k in ("n", "final", "final_freq_weighted", "kaznlp", "apertium")},
                     ensure_ascii=False, indent=2))
    for p in per:
        if p["final"] == "owner":
            print(f"  {p['form']:<24} {p['regex_split']:<28} kaznlp={p['kaznlp']:<14} apertium={p['apertium']:<12}"
                  f" {p['apertium_analyses'][:2]}")


if __name__ == "__main__":
    main()

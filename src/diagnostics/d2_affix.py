"""D2 — Affix Isolation Rate.

For every Kazakh negative verb form (stem + neg affix + ending) check whether
the XLM-R / e5 tokenizer puts a token boundary before and after the negative
affix. Isolation = both boundaries; "any" = at least one.

Controls (Russian):
  * separate particle «не» — share of «не» that is its own token;
  * fused prefix «не-» (неуплата) — share with a boundary right after «не».

Input: data/diagnostics/d2_kz_neg_forms.tsv (form, stem, affix, ending, source, status)

Usage:
    python -m src.diagnostics.d2_affix
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["xlm-roberta-base", "intfloat/multilingual-e5-base"]

RU_PARTICLE_PHRASES = [
    "не вправе", "не допускается", "не может", "не подлежит", "не облагается", "не обязан",
    "не признается", "не менее", "не позднее", "не более", "не вправе требовать", "не включается",
]
# Fused negative prefix «не-»: word = «не» + rest. Hand-picked from the Labor Code.
RU_FUSED = [
    "неисполнение", "ненадлежащее", "непредоставление", "неуплата", "невыплата", "нетрудоспособности",
    "несоблюдение", "неявка", "невозможно", "недопустимо", "неполный", "неправомерный", "незаконный",
    "несовершеннолетний", "неоплачиваемый", "невыполнение", "неприменение", "необоснованный",
    "неразглашение", "неконкуренции", "несогласие", "неосуществление", "непрерывный", "неработающий",
    "непригодный", "неисправный", "неудовлетворительный", "недееспособный", "несвоевременный",
    "неблагоприятных",
]


def boundaries(tok, word):
    """Character offsets inside `word` where a token starts or ends."""
    enc = tok(word, add_special_tokens=False, return_offsets_mapping=True)
    b = set()
    for s, e in enc["offset_mapping"]:
        if e > s:
            b.add(s)
            b.add(e)
    return b, tok.convert_ids_to_tokens(enc["input_ids"])


def load_forms(path, include_unchecked):
    # strip: editors may pad TSV columns with spaces when aligning them
    rows = [{k.strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")]
    if not include_unchecked:
        # ok = owner, ok_auto = confirmed by kaznlp + apertium (src/diagnostics/d2_crosscheck.py)
        rows = [r for r in rows if r["status"] in ("ok", "ok_auto")]
    return rows


def analyse_kz(tok, rows):
    per, n_before, n_after, n_both, n_single = [], 0, 0, 0, 0
    for r in rows:
        form = r["form"]
        a = len(r["stem"])
        b = a + len(r["affix"])
        assert form[a:b] == r["affix"], r
        bnd, pieces = boundaries(tok, form)
        before, after = a in bnd, b in bnd
        single = any(p.lstrip("▁") == r["affix"] for p in pieces) and before and after
        n_before += before
        n_after += after
        n_both += before and after
        n_single += single
        per.append({"form": form, "split": f"{r['stem']}+{r['affix']}+{r['ending']}", "tokens": pieces,
                    "boundary_before": before, "boundary_after": after})
    n = len(rows)
    # Pattern of the affix inside the tokenization; "buried" = no boundary on either side.
    cats = {"isolated": 0, "starts_token": 0, "ends_token": 0, "buried": 0}
    cats_w = dict.fromkeys(cats, 0)
    freq = [int(r.get("freq") or 1) for r in rows]
    for p, f in zip(per, freq):
        b, a = p["boundary_before"], p["boundary_after"]
        k = "isolated" if b and a else "starts_token" if b else "ends_token" if a else "buried"
        p["pattern"] = k
        cats[k] += 1
        cats_w[k] += f
    total_f = sum(freq)
    return {
        "n": n,
        "isolation_rate": n_both / n,
        "any_boundary_rate": (sum(1 for p in per if p["boundary_before"] or p["boundary_after"])) / n,
        "boundary_before_rate": n_before / n,
        "boundary_after_rate": n_after / n,
        "affix_single_token_rate": n_single / n,
        "patterns": {k: v / n for k, v in cats.items()},
        "corpus_tokens": total_f,
        "isolation_rate_freq_weighted": cats_w["isolated"] / total_f,
        "patterns_freq_weighted": {k: v / total_f for k, v in cats_w.items()},
    }, per


def analyse_ru(tok):
    sep = []
    for ph in RU_PARTICLE_PHRASES:
        pieces = tok.convert_ids_to_tokens(tok(ph, add_special_tokens=False)["input_ids"])
        sep.append({"phrase": ph, "tokens": pieces, "ne_own_token": pieces[0] == "▁не"})
    fused = []
    for w in RU_FUSED:
        bnd, pieces = boundaries(tok, w)
        fused.append({"word": w, "tokens": pieces, "boundary_after_ne": 2 in bnd})
    return {
        "particle_ne_own_token_rate": sum(x["ne_own_token"] for x in sep) / len(sep),
        "particle_n": len(sep),
        "fused_prefix_boundary_rate": sum(x["boundary_after_ne"] for x in fused) / len(fused),
        "fused_n": len(fused),
    }, sep, fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default=ROOT / "data/diagnostics/d2_kz_neg_forms.tsv")
    ap.add_argument("--include-unchecked", action="store_true",
                    help="also use forms not yet verified by the owner (status != ok)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_forms(args.forms, args.include_unchecked)
    toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}
    same_vocab = toks[MODELS[0]].get_vocab() == toks[MODELS[1]].get_vocab()

    out = {
        "diagnostic": "D2_affix_isolation",
        "config": {"models": MODELS, "forms_file": str(Path(args.forms).relative_to(ROOT)),
                   "include_unchecked": args.include_unchecked, "date": date.today().isoformat()},
        "tokenizers_identical_vocab": same_vocab,
        "by_model": {},
    }
    for m, tok in toks.items():
        kz, per = analyse_kz(tok, rows)
        ru, sep, fused = analyse_ru(tok)
        by_source = {}
        for src in sorted({r["source"] for r in rows}):
            sub = [r for r in rows if r["source"] == src]
            by_source[src] = analyse_kz(tok, sub)[0]
        by_affix = {}
        for aff in sorted({r["affix"] for r in rows}):
            sub = [r for r in rows if r["affix"] == aff]
            by_affix[aff] = {"n": len(sub), "isolation_rate": analyse_kz(tok, sub)[0]["isolation_rate"]}
        out["by_model"][m] = {"kz": kz, "kz_by_source": by_source, "kz_by_affix": by_affix,
                              "ru_control": ru, "kz_forms": per, "ru_particle": sep, "ru_fused": fused}

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    name = "d2_affix_isolation" + (f"_{args.tag}" if args.tag else "") + ".json"
    (res / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("tokenizers identical vocab:", same_vocab)
    for m, d in out["by_model"].items():
        print(f"\n== {m}")
        print(json.dumps({"kz": d["kz"], "ru_control": d["ru_control"]}, indent=2))
        print("examples:")
        for p in d["kz_forms"][:10]:
            print(f"  {p['split']:<28} -> {' '.join(p['tokens'])}")


if __name__ == "__main__":
    main()

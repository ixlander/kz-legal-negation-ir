"""NLI training and validation data for the reranker.

    xnli_{ru,tr}     stratified subsample of XNLI train (H5: with and without Turkish)
    snli_kk          stratified subsample of machine-translated SNLI-kk (extra H5 variant)
    kardes_kk_*      Kardeş-NLU Kazakh XNLI dev/test (post-edited by native speakers), validation only
    synthetic_{ru,kz} rule-based pairs from the distractor codes (never from Labor/Tax, so the
                     benchmark articles do not leak into training)

Synthetic pairs, one rewrite of one legal sentence each:
    contradiction  polarity flip («вправе» -> «не вправе») or modality swap («допускается» ->
                   «запрещается»), not only an inserted «не»
    entailment     a negated cue that keeps the meaning: range negation («не менее» -> «не меньше»)
                   and synonymous negatives («запрещается» -> «не допускается»); these
                   counter-examples (H6) teach that «не» alone does not mean contradiction
    neutral        a tighter numeric bound («не менее трех» -> «не менее пяти»), lexically almost the
                   premise; another sentence of the same article fills at most a third of the neutrals
    numbers        a looser numeric bound is an entailment («не менее трех» -> «не менее двух»)
Pairs are split train / test by article (10% of articles held out).
Kazakh synthetic text is rule-made, not human-checked: status NEEDS_KZ_CHECK.

Labels: 0 entailment, 1 neutral, 2 contradiction.

Usage:
    python -m src.rerank.nli_data --per-lang 50000 --synthetic 4000
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT, OUT, CORPUS = ROOT / "data/external", ROOT / "data/nli", ROOT / "data/corpus"
SEED = 42
ENT, NEU, CON = 0, 1, 2
BENCHMARK_CODES = {"labor", "tax"}
KARDES_LABEL = {"entailment": ENT, "neutral": NEU, "contradiction": CON}

# (pattern, replacement, label, kind); patterns match whole words in a lowercased sentence
RU_RULES = [
    (r"\bне вправе\b", "вправе", CON, "flip"),
    (r"(?<!не )\bвправе\b", "не вправе", CON, "flip"),
    (r"\bне допускается\b", "допускается", CON, "flip"),
    (r"(?<!не )\bдопускается\b", "запрещается", CON, "modality"),
    (r"\bзапрещается\b", "допускается", CON, "modality"),
    (r"(?<!не )\bобязан\b", "не обязан", CON, "flip"),
    (r"\bне подлежит\b", "подлежит", CON, "flip"),
    (r"(?<!не )\bподлежит\b", "не подлежит", CON, "flip"),
    (r"(?<!не )\bявляется\b", "не является", CON, "flip"),
    (r"(?<!не )\bпризнается\b", "не признается", CON, "flip"),
    (r"(?<!не )\bможет\b", "не может", CON, "flip"),
    (r"\bимеет право\b", "не имеет права", CON, "flip"),
    (r"\bзапрещается\b", "не допускается", ENT, "synonym_neg"),
    (r"\bне допускается\b", "запрещается", ENT, "synonym_neg"),
    # range rewrites keep the genitive of the following numeral, so hypotheses stay grammatical
    (r"\bне менее\b", "не меньше", ENT, "range"),
    (r"\bне более\b", "не свыше", ENT, "range"),
    (r"\bне ранее\b", "не раньше", ENT, "range"),
    (r"\bне позднее\b", "не позже", ENT, "range"),
    (r"\bне менее\b", "менее", CON, "range"),
    (r"\bне более\b", "более", CON, "range"),
]
KZ_RULES = [
    (r"\bқұқылы емес\b", "құқылы", CON, "flip"),
    (r"\bқұқылы\b(?! емес)", "құқылы емес", CON, "flip"),
    (r"\bжол берілмейді\b", "жол беріледі", CON, "flip"),
    (r"\bжол беріледі\b", "жол берілмейді", CON, "flip"),
    (r"\bтыйым салынады\b", "рұқсат етіледі", CON, "modality"),
    (r"\bміндетті\b(?! емес)", "міндетті емес", CON, "flip"),
    (r"\bжатады\b", "жатпайды", CON, "flip"),
    (r"\bжатпайды\b", "жатады", CON, "flip"),
    (r"\bболып табылады\b", "болып табылмайды", CON, "flip"),
    (r"\bболып табылмайды\b", "болып табылады", CON, "flip"),
    (r"\bтыйым салынады\b", "жол берілмейді", ENT, "synonym_neg"),
    (r"\bжол берілмейді\b", "тыйым салынады", ENT, "synonym_neg"),
    # range: only flips that keep the word order («бір жылдан кем емес» -> «бір жылдан кем»)
    (r"\bкем емес\b", "кем", CON, "range"),
    (r"\bаспауға тиіс\b", "асуға тиіс", CON, "range"),
]
SENTENCE = re.compile(r"(?<=[.;])\s+|\n+")


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name:20s} {len(rows):>6d}  " + str(pd.Series([r['label'] for r in rows]).value_counts().sort_index().to_dict()))


def stratified(df, n, seed=SEED):
    per = n // 3
    return pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in df.groupby("label")]).sample(
        frac=1, random_state=seed)


def external(n):
    for lang in ("ru", "tr"):
        df = pd.read_parquet(EXT / f"xnli/{lang}/train-00000-of-00001.parquet")
        df = stratified(df[df.label.isin([ENT, NEU, CON])], n)
        write(f"xnli_{lang}", [{"premise": p, "hypothesis": h, "label": int(l), "source": f"xnli-{lang}"}
                               for p, h, l in zip(df.premise, df.hypothesis, df.label)])
    df = pd.read_parquet(EXT / "snli-kazakh/train.parquet")
    df["label"] = df.label.astype(int)
    df = stratified(df[df.premise_kk.str.len().gt(0) & df.hypothesis_kk.str.len().gt(0)], n)
    write("snli_kk", [{"premise": p, "hypothesis": h, "label": int(l), "source": "snli-kk"}
                      for p, h, l in zip(df.premise_kk, df.hypothesis_kk, df.label)])
    for split in ("dev", "test"):
        df = pd.read_csv(EXT / f"kardes-nlu/Data/kazakh/xnli.{split}.kk.csv")
        write(f"kardes_kk_{split}", [{"premise": p, "hypothesis": h, "label": KARDES_LABEL[g], "source": "kardes-kk"}
                                     for p, h, g in zip(df.s1_translation, df.s2_translation, df.gold_label)
                                     if g in KARDES_LABEL])


def sentences(lang):
    """(article key, sentence) from the distractor codes; 8-60 words, no tables or lists."""
    out = []
    for p in sorted(CORPUS.glob(f"*_{lang}.jsonl")):
        if p.stem.rsplit("_", 1)[0] in BENCHMARK_CODES:
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            art = f"{r['code']}:{r['article_no']}"
            for s in SENTENCE.split(r["text"]):
                s = re.sub(r"^\s*\d+(-\d+)?[.)]\s*", "", s).strip()
                if 8 <= len(s.split()) <= 60:
                    out.append((art, s))
    return out


# Numerals that can be swapped without changing the grammar around them: Ru genitive forms after
# «не менее / не более» (one -> «одного» is left out, it takes a singular noun); Kz numerals do not inflect.
RU_NUM = {"двух": 2, "трех": 3, "четырех": 4, "пяти": 5, "шести": 6, "семи": 7, "восьми": 8, "девяти": 9,
          "десяти": 10, "двенадцати": 12, "пятнадцати": 15, "двадцати": 20, "тридцати": 30, "шестидесяти": 60,
          "девяноста": 90, "ста": 100}
KZ_NUM = {"екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7, "сегіз": 8, "тоғыз": 9, "он": 10,
          "жиырма": 20, "отыз": 30, "алпыс": 60, "тоқсан": 90, "жүз": 100}
# (pattern with group "num", bound): "min" = at least the number, "max" = at most the number
NUM_PATTERNS = {
    "ru": [(r"\bне (?:менее|ранее) (?P<num>{n})\b", "min"), (r"\bне (?:более|позднее) (?P<num>{n})\b", "max")],
    "kz": [(r"(?<!\S)(?P<num>{n}) \S+ кем емес\b", "min"), (r"\bкемінде (?P<num>{n})(?!\S)", "min"),
           (r"(?<!\S)(?P<num>{n}) \S+ (?:аспауға тиіс|аспайтын|аспайды)\b", "max")],
}
TEST_SHARE = 0.1


def numeric_rewrites(s, lang, rng):
    """Change the bound of a range: a looser bound is entailed, a tighter one is neutral.
    «не менее трех дней» -> «не менее двух дней» (entailment) / «не менее пяти дней» (neutral)."""
    words = RU_NUM if lang == "ru" else KZ_NUM
    alt = "|".join(sorted(map(re.escape, words), key=len, reverse=True))
    for pat, bound in NUM_PATTERNS[lang]:
        m = re.search(pat.format(n=alt), s)
        if not m:
            continue
        before, after = s[:m.start("num")].split()[-1:], s[m.end("num"):].split()[:1]
        if any(w.lower() in words for w in before + after):
            return []   # compound numeral («он бес», «двадцати пяти»): one word cannot be swapped safely
        x = words[m.group("num")]
        smaller = [w for w, v in words.items() if v < x]
        larger = [w for w, v in words.items() if v > x]
        looser, tighter = (smaller, larger) if bound == "min" else (larger, smaller)
        out = []
        for label, choices in ((ENT, looser), (NEU, tighter)):
            if choices:
                hyp = s[:m.start("num")] + rng.choice(choices) + s[m.end("num"):]
                out.append((label, hyp))
        return out
    return []


def is_test(article):
    """Held-out articles, so the synthetic test split never shares a premise with training."""
    return int(hashlib.md5(article.encode()).hexdigest(), 16) % 100 < TEST_SHARE * 100


def synthetic(lang, n, seed=SEED):
    rng = random.Random(seed)
    rules = RU_RULES if lang == "ru" else KZ_RULES
    sents = sentences(lang)
    by_article = {}
    for art, s in sents:
        by_article.setdefault(art, []).append(s)
    rng.shuffle(sents)

    pool = {CON: [], ENT: [], NEU: []}
    for art, s in sents:
        low = s[0].lower() + s[1:]
        for pat, rep, label, kind in rng.sample(rules, len(rules)):
            new, k = re.subn(pat, rep, low, count=1)
            if k:
                pool[label].append({"premise": s, "hypothesis": new[0].upper() + new[1:], "label": label,
                                    "kind": kind, "article": art})
                break
        for label, hyp in numeric_rewrites(s, lang, rng):
            pool[label].append({"premise": s, "hypothesis": hyp, "label": label, "kind": "number", "article": art})
    # lexically distant neutrals only fill what the close (numeric) ones cannot, at most a third
    distant = [{"premise": s, "hypothesis": rng.choice(o), "label": NEU, "kind": "same_article", "article": art}
               for art, s in sents for o in [[x for x in by_article[art] if x != s]] if o]
    quota = {CON: int(n * 0.4), ENT: int(n * 0.3), NEU: n - int(n * 0.4) - int(n * 0.3)}
    close_neu = rng.sample(pool[NEU], min(quota[NEU], len(pool[NEU])))
    fill = min(quota[NEU] - len(close_neu), quota[NEU] // 3)
    rows = (rng.sample(pool[CON], min(quota[CON], len(pool[CON])))
            + rng.sample(pool[ENT], min(quota[ENT], len(pool[ENT])))
            + close_neu + rng.sample(distant, max(fill, 0)))
    rng.shuffle(rows)
    for r in rows:
        r["source"] = f"synthetic-{lang}"
        r["split"] = "test" if is_test(r["article"]) else "train"
        if lang == "kz":
            r["status"] = "NEEDS_KZ_CHECK"
    write(f"synthetic_{lang}", rows)
    print("  kinds:", pd.Series([r["kind"] for r in rows]).value_counts().to_dict(),
          " test:", sum(r["split"] == "test" for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=50_000)
    ap.add_argument("--synthetic", type=int, default=4_000)
    a = ap.parse_args()
    external(a.per_lang)
    for lang in ("ru", "kz"):
        synthetic(lang, a.synthetic)


if __name__ == "__main__":
    main()

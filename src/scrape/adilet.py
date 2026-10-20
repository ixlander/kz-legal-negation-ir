"""Download codes from adilet.zan.kz and parse them into aligned Ru/Kz chunks.

The site is a JS app, but it loads the act from a public JSON endpoint
(/api/documents/by-ngr/<NGR>?language=rus|kaz) with the full HTML in
`document.text_content`, so no browser is needed.

Chunk = article; an article longer than MAX_WORDS (in either language) is split
by its numbered points («1.», «2.» ...). The split decision is taken for the
Ru/Kz pair jointly, so `chunk_id`s stay aligned across languages.

Usage:
    python -m src.scrape.adilet --code labor
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/adilet"
CORPUS = ROOT / "data/corpus"
API = "https://adilet.zan.kz/api/documents/by-ngr/{ngr}?language={lang}"
CODES = {"labor": "K1500000414", "tax": "K2500000214"}
LANGS = {"ru": "rus", "kz": "kaz"}
MAX_WORDS = 300

HEAD_RE = {
    "ru": re.compile(r"^Статья\s+(\d+(?:-\d+)?)\.\s*(.*)$"),
    "kz": re.compile(r"^(\d+(?:-\d+)?)-бап\.\s*(.*)$"),
}
EXCLUDED_RE = re.compile(r"(исключен|алынып таста)", re.I)
# Footnotes that are not wrapped in span.note: a whole paragraph or a tail of one.
NOTE_RE = re.compile(r"(?:^|\s)(?:Сноска|Ескерту)\.\s.*$", re.S)
HEADING_RE = re.compile(r"^(?:Глава|РАЗДЕЛ|Параграф)\s+\d|^\d+-(?:тарау|БӨЛІМ|бөлім|параграф)\b")
POINT_RE = re.compile(r"^(\d+(?:-\d+)?)\.\s")


def fetch(code, lang):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{code}_{LANGS[lang]}.json"
    if not path.exists():
        req = urllib.request.Request(API.format(ngr=CODES[code], lang=LANGS[lang]),
                                     headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return json.loads(path.read_text(encoding="utf-8"))["document"]


def _clean(s):
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


_MIXED_I = re.compile(r"\w*[а-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]\w*")


def normalize_kz(s):
    """adilet Kazakh text sometimes has Latin i/I inside Cyrillic words (белгiленбесе)."""
    return _MIXED_I.sub(lambda m: m.group(0).replace("i", "і").replace("I", "І"), s)


def parse_articles(html, lang):
    """-> list of {article_no, title, excluded, points: [(para_no|None, text)]}"""
    soup = BeautifulSoup(html, "lxml")
    for el in soup.select("span.note, table"):  # footnotes («Сноска»/«Ескерту») and tables are not indexed
        el.decompose()
    arts, cur = [], None
    for p in soup.find_all(["p", "h3"]):
        if p.name == "h3":
            continue
        text = _clean(p.get_text(" "))
        if lang == "kz":
            text = normalize_kz(text)
        if not text:
            continue
        b = p.find("b")
        btext = _clean(b.get_text(" ")) if b else ""
        if lang == "kz":
            btext = normalize_kz(btext)
        m = HEAD_RE[lang].match(text) if b and btext == text else None
        if m:
            cur = {"article_no": m.group(1), "title": m.group(2), "points": [],
                   "excluded": bool(EXCLUDED_RE.search(m.group(2)))}
            arts.append(cur)
            continue
        if cur is None:
            continue
        text = NOTE_RE.sub("", text).strip()
        if not text or HEADING_RE.match(text):  # a chapter heading in a <p> (seen once: «Глава 18.»)
            continue
        pm = POINT_RE.match(text)
        if pm:
            cur["points"].append([pm.group(1), text])
        elif cur["points"]:
            cur["points"][-1][1] += "\n" + text
        else:
            cur["points"].append([None, text])
    for a in arts:
        if a["points"] and all(EXCLUDED_RE.search(t) and len(t.split()) < 40 for _, t in a["points"]):
            a["excluded"] = True
    return arts


def build_chunks(code, arts_by_lang):
    idx = {lang: {a["article_no"]: a for a in arts} for lang, arts in arts_by_lang.items()}
    shared = [n for n in idx["ru"] if n in idx["kz"]]
    out = {lang: [] for lang in idx}
    stats = {"articles_ru": len(idx["ru"]), "articles_kz": len(idx["kz"]), "aligned": len(shared),
             "excluded": 0, "split": 0, "only_ru": sorted(set(idx["ru"]) - set(idx["kz"])),
             "only_kz": sorted(set(idx["kz"]) - set(idx["ru"]))}
    for n in shared:
        ru, kz = idx["ru"][n], idx["kz"][n]
        if ru["excluded"] or kz["excluded"] or not ru["points"] or not kz["points"]:
            stats["excluded"] += 1
            continue
        words = max(sum(len(t.split()) for _, t in a["points"]) for a in (ru, kz))
        same_points = [p for p, _ in ru["points"]] == [p for p, _ in kz["points"]]
        split = words > MAX_WORDS and same_points and len(ru["points"]) > 1
        stats["split"] += split
        for lang, a in (("ru", ru), ("kz", kz)):
            if split:
                parts = [(p or "0", t) for p, t in a["points"]]
            else:
                parts = [(None, "\n".join(t for _, t in a["points"]))]
            for para_no, text in parts:
                cid = f"{code}:{n}" + (f":{para_no}" if para_no else "")
                out[lang].append({"chunk_id": cid, "code": code, "article_no": n, "para_no": para_no,
                                  "lang": lang, "title": a["title"], "text": text})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=list(CODES), default="labor")
    args = ap.parse_args()

    docs = {lang: fetch(args.code, lang) for lang in LANGS}
    arts = {lang: parse_articles(d["text_content"], lang) for lang, d in docs.items()}
    chunks, stats = build_chunks(args.code, arts)
    stats["edition"] = {lang: {"updated_at": d["updated_at"], "end_date": d["end_date"],
                               "is_actual": d["is_actual"]} for lang, d in docs.items()}
    CORPUS.mkdir(parents=True, exist_ok=True)
    for lang, rows in chunks.items():
        with open(CORPUS / f"{args.code}_{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"chunks_{lang}"] = len(rows)
    (CORPUS / f"{args.code}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

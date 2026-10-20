"""Download codes from adilet.zan.kz and parse them into aligned Ru/Kz chunks.

The site is a JS app, but it loads the act from a public JSON endpoint
(/api/documents/by-ngr/<NGR>?language=rus|kaz) with the full HTML in
`document.text_content`, so no browser is needed.

Chunk = point of an article («1.», «2.» ...) when the point numbering is the same in
Ru and Kz, otherwise the whole article. A chunk still longer than MAX_WORDS (in either
language) is split into groups of consecutive sub-points («1)», «2)» ...) when the
sub-point numbering matches. All split decisions are taken for the Ru/Kz pair jointly,
so `chunk_id`s stay aligned: code:article[:point][#group]. The article title is kept in
`title` and prepended at indexing time (CLAUDE.md §12, 21.10.2026).

Usage:
    python -m src.scrape.adilet --code labor
    python -m src.scrape.adilet --all
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
# Codes in force with the same Ru/Kz edition (data/corpus/codes_inventory.md); the Entrepreneurial
# Code is left out: its Ru and Kz editions differ. Questions are asked only on labor and tax.
CODES = {
    "labor": "K1500000414", "tax": "K2500000214",
    "civil_general": "K940001000_", "civil_special": "K990000409_", "civil_procedure": "K1500000377",
    "criminal": "K1400000226", "criminal_procedure": "K1400000231", "penal": "K1400000234",
    "admin_offences": "K1400000235", "admin_procedure": "K2000000350", "family": "K1100000518",
    "land": "K030000442_", "forest": "K030000477_", "water": "K2500000178", "ecology": "K2100000400",
    "subsoil": "K1700000125", "customs": "K1700000123", "budget": "K2500000171", "health": "K2000000360",
    "social": "K2300000224", "construction": "K2600000253", "digital": "K2600000255",
}
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
    """`code` is a short name from CODES or an adilet NGR (e.g. K1500000414)."""
    RAW.mkdir(parents=True, exist_ok=True)
    ngr = CODES.get(code, code)
    path = RAW / f"{ngr}_{LANGS[lang]}.json"
    if not path.exists():
        req = urllib.request.Request(API.format(ngr=ngr, lang=LANGS[lang]),
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
        text = _clean(p.get_text(" "))
        if lang == "kz":
            text = normalize_kz(text)
        if not text:
            continue
        if p.name == "h3":
            # chapters/sections; in some acts (Kz Tax Code) also the article headings
            m = HEAD_RE[lang].match(text)
            if not m:
                continue
        else:
            # the heading may be split over several <b> (APPK: «<b>35-бап. </b><b>Аудармашы</b>»)
            bs = p.find_all("b")
            btext = _clean(" ".join(b.get_text(" ") for b in bs))
            if lang == "kz":
                btext = normalize_kz(btext)
            m = HEAD_RE[lang].match(text) if bs and btext == text else None
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


SUBPOINT_RE = re.compile(r"^(\d+(?:-\d+)?)\)\s")


def _words(t):
    return len(t.split())


def _subpoints(text):
    """'intro\\n1) a\\n2) b' -> [[None, 'intro'], ['1', '1) a'], ['2', '2) b']]; other lines stay with their sub-point."""
    out = []
    for line in text.split("\n"):
        m = SUBPOINT_RE.match(line)
        if m or not out:
            out.append([m.group(1) if m else None, line])
        else:
            out[-1][1] += "\n" + line
    return out


def _group_subpoints(ru, kz):
    """Split a long Ru/Kz pair into groups of consecutive sub-points, with the same boundaries in both."""
    sr, sk = _subpoints(ru), _subpoints(kz)
    if [m for m, _ in sr] != [m for m, _ in sk] or len(sr) < 3:
        return [(ru, kz)]
    groups, cur, wr, wk = [], [], 0, 0
    for (_, a), (_, b) in zip(sr, sk):
        if cur and max(wr + _words(a), wk + _words(b)) > MAX_WORDS:
            groups.append(cur)
            cur, wr, wk = [], 0, 0
        cur.append((a, b))
        wr, wk = wr + _words(a), wk + _words(b)
    groups.append(cur)
    return [("\n".join(a for a, _ in g), "\n".join(b for _, b in g)) for g in groups]


def build_chunks(code, arts_by_lang):
    idx = {lang: {a["article_no"]: a for a in arts} for lang, arts in arts_by_lang.items()}
    shared = [n for n in idx["ru"] if n in idx["kz"]]
    out = {lang: [] for lang in idx}
    stats = {"articles_ru": len(idx["ru"]), "articles_kz": len(idx["kz"]), "aligned": len(shared),
             "excluded": 0, "by_point": 0, "whole_article": 0, "subpoint_groups": 0,
             "over_max_words": 0, "only_ru": sorted(set(idx["ru"]) - set(idx["kz"])),
             "only_kz": sorted(set(idx["kz"]) - set(idx["ru"]))}
    for n in shared:
        ru, kz = idx["ru"][n], idx["kz"][n]
        if ru["excluded"] or kz["excluded"] or not ru["points"] or not kz["points"]:
            stats["excluded"] += 1
            continue
        same_points = [p for p, _ in ru["points"]] == [p for p, _ in kz["points"]]
        if same_points and len(ru["points"]) > 1:
            units = [(p or "0", tr, tk) for (p, tr), (_, tk) in zip(ru["points"], kz["points"])]
            stats["by_point"] += 1
        else:
            units = [(None, "\n".join(t for _, t in ru["points"]), "\n".join(t for _, t in kz["points"]))]
            stats["whole_article"] += 1
        for para_no, tr, tk in units:
            base = f"{code}:{n}" + (f":{para_no}" if para_no else "")
            parts = _group_subpoints(tr, tk) if max(_words(tr), _words(tk)) > MAX_WORDS else [(tr, tk)]
            stats["subpoint_groups"] += len(parts) if len(parts) > 1 else 0
            for g, (pr, pk) in enumerate(parts, 1):
                cid = base + (f"#{g}" if len(parts) > 1 else "")
                stats["over_max_words"] += max(_words(pr), _words(pk)) > MAX_WORDS
                for lang, a, text in (("ru", ru, pr), ("kz", kz, pk)):
                    out[lang].append({"chunk_id": cid, "code": code, "article_no": n, "para_no": para_no,
                                      "lang": lang, "title": a["title"], "text": text})
    return out, stats


def build(code):
    docs = {lang: fetch(code, lang) for lang in LANGS}
    arts = {lang: parse_articles(d["text_content"], lang) for lang, d in docs.items()}
    chunks, stats = build_chunks(code, arts)
    stats["ngr"] = CODES[code]
    stats["edition"] = {lang: {"updated_at": d["updated_at"], "end_date": d["end_date"],
                               "is_actual": d["is_actual"]} for lang, d in docs.items()}
    CORPUS.mkdir(parents=True, exist_ok=True)
    for lang, rows in chunks.items():
        with open(CORPUS / f"{code}_{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"chunks_{lang}"] = len(rows)
    (CORPUS / f"{code}_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=list(CODES), default="labor")
    ap.add_argument("--all", action="store_true", help="build every code in CODES")
    args = ap.parse_args()
    for code in (CODES if args.all else [args.code]):
        st = build(code)
        print(f"{code:<20} articles {st['aligned']:>5}  chunks {st['chunks_ru']:>6}  by_point {st['by_point']:>4}  "
              f"whole {st['whole_article']:>4}  groups {st['subpoint_groups']:>4}  >max {st['over_max_words']:>3}")


if __name__ == "__main__":
    main()

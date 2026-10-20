"""Inventory of the codes of the Republic of Kazakhstan in force, Ru + Kz, from adilet.zan.kz.

Candidates come from the site's public search (POST /api/search, document_type == «Кодекс»,
status != yts = repealed), minus codes replaced by a newer code of the same name. For each
we download both languages and report editions, article counts and Ru/Kz article alignment.
Nothing is added to the corpus here.

Usage:
    python -m src.scrape.list_codes   -> data/corpus/codes_inventory.{json,md}
"""
import json
import urllib.request
from pathlib import Path

from src.scrape.adilet import fetch, parse_articles

ROOT = Path(__file__).resolve().parents[2]
SEARCH = "https://adilet.zan.kz/api/search"
QUERIES = ["кодекс", "Кодекс Республики Казахстан", "кодексі"]
# in force formally, but replaced by a newer code (new Tax Code 2025, Budget Code 2025)
SUPERSEDED = {"K1700000120", "T1800000120"}


def search(q, page):
    body = json.dumps({"query": q, "filters": {}, "page": page, "page_size": 100}).encode()
    req = urllib.request.Request(SEARCH, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def candidates():
    found = {}
    for q in QUERIES:
        for page in range(1, 8):
            for x in search(q, page)["results"]:
                x = x.get("document", x)
                if x.get("document_type") == "Кодекс" and x.get("status") != "yts":
                    found.setdefault(x["group_number"], x)
    return {g: x for g, x in found.items() if g not in SUPERSEDED}


def main():
    inv = []
    for ngr, meta in sorted(candidates().items()):
        row = {"ngr": ngr, "title_ru": meta.get("title_ru", "")}
        try:
            docs = {l: fetch(ngr, l) for l in ("ru", "kz")}
        except Exception as e:  # a language version may be missing
            row["error"] = str(e)
            inv.append(row)
            continue
        arts = {l: parse_articles(d["text_content"], l) for l, d in docs.items()}
        n = {l: {a["article_no"] for a in arts[l] if not a["excluded"]} for l in arts}
        row.update({
            "title_kz": docs["kz"].get("title_kz") or docs["kz"].get("title_ru", ""),
            "end_date": {l: d["end_date"][:10] for l, d in docs.items()},
            "updated": {l: d["updated_at"][:10] for l, d in docs.items()},
            "same_edition": docs["ru"]["end_date"] == docs["kz"]["end_date"],
            "articles": {l: len(n[l]) for l in n},
            "aligned_articles": len(n["ru"] & n["kz"]),
            "words": {l: sum(len(t.split()) for a in arts[l] for _, t in a["points"]) for l in arts},
        })
        inv.append(row)
        print(ngr, row["articles"], row["aligned_articles"], row["same_edition"], row["title_ru"][:60])

    out = ROOT / "data/corpus/codes_inventory"
    out.with_suffix(".json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| NGR | Code | Articles Ru / Kz | Aligned | Words Ru | Same edition | Updated Ru / Kz |",
             "|---|---|---|---|---|---|---|"]
    for r in inv:
        if "error" in r:
            lines.append(f"| {r['ngr']} | {r['title_ru']} | error: {r['error'][:40]} | | | | |")
            continue
        lines.append(f"| {r['ngr']} | {r['title_ru']} | {r['articles']['ru']} / {r['articles']['kz']} | "
                     f"{r['aligned_articles']} | {r['words']['ru']:,} | {'yes' if r['same_edition'] else 'NO'} | "
                     f"{r['updated']['ru']} / {r['updated']['kz']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

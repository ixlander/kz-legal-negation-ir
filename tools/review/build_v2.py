"""Build the two review pages of the second annotation round.

    main   drafts of the new questions (text, gold, twin) for the first annotator
    blind  for a second annotator who has seen no drafts and no model labels:
           relevance of pooled search results for every final question, and the NLI pairs

The relevance pool of a question is the union of the top results of several systems in both
languages plus its gold and twin, shuffled, with nothing that tells them apart.

Usage:
    python tools/review/build_v2.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
POOL_RUNS = ["bge-m3/B0", "e5-base/B0", "bge-m3/B2", "bge-m3/B3'-bge-reranker-v2-m3"]
POOL_DEPTH, SEED = 3, 42
PAGES = {
    "main": {"title": "Вопросы v2 Ru/Kz", "sections": "sections_v2_main.js"},
    "blind": {"title": "Слепая разметка норм", "sections": "sections_v2_blind.js"},
}


def jl(path):
    return [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]


def corpus(lang):
    return {r["chunk_id"]: r for code in ("labor", "tax") for r in jl(f"data/corpus/{code}_{lang}.jsonl")}


def card_id(chunk_id):
    return re.sub(r"[^A-Za-z0-9_.~:@+-]", "_", chunk_id)


def chunk(c, ru, kz):
    return {"id": c, "title": ru[c]["title"], "text": ru[c]["text"][:1800],
            "title_kz": kz[c]["title"], "text_kz": kz[c]["text"][:1800]}


def main_data(ru, kz):
    rows = jl("data/eval/queries_v2_draft.jsonl")
    kz_q = {r["pair_id"]: r["question"] for r in rows if r["lang"] == "kz"}
    return {"questions": [
        {"key": "q-" + r["qid"], "qid": r["qid"], "slice": r["slice"], "cell": r["cell"], "code": r["pair_id"][:3],
         "question": r["question"], "question_kz": kz_q[r["pair_id"]],
         "gold": [chunk(c, ru, kz) for c in r["gold_ids"]], "twin": [chunk(c, ru, kz) for c in r["twin_ids"]]}
        for r in rows if r["lang"] == "ru"]}


def blind_data(ru, kz):
    queries = jl("data/eval/queries.jsonl")
    pool = {q["pair_id"]: set(q["gold_ids"]) | set(q["twin_ids"]) for q in queries}
    for split in ("dev", "test"):
        res = json.loads((ROOT / f"results/final_ablation_{split}.json").read_text(encoding="utf-8"))
        for L in res["langs"].values():
            for run in POOL_RUNS:
                for r in L["per_query"][run]:
                    pool[r["qid"][3:]] |= {c for c in r["top5"][:POOL_DEPTH] if c in ru and c in kz}
    question = {(q["pair_id"], q["lang"]): q["question"] for q in queries}
    rng = random.Random(SEED)
    cards = []
    pairs = sorted(pool)
    for n, pair in enumerate(pairs, 1):
        chunks = sorted(pool[pair])
        rng.shuffle(chunks)
        for c in chunks:
            cards.append({"key": f"rel-{pair}-{card_id(c)}", "pair_id": pair, "n": n, "of": len(pairs),
                          "question": question[pair, "ru"], "question_kz": question[pair, "kz"],
                          "chunk": chunk(c, ru, kz)})
    by_pair = {}
    for r in jl("data/nli/val_manual.jsonl"):
        if r.get("status") == "ok":
            by_pair.setdefault(r["pair_id"], {})[r["lang"]] = r
    nli = [{"key": f"nlib-{pid}", "pair_id": pid, "premise_ru": p["ru"]["premise"], "hypothesis_ru": p["ru"]["hypothesis"],
            "premise_kz": p["kz"]["premise"], "hypothesis_kz": p["kz"]["hypothesis"]}
           for pid, p in sorted(by_pair.items()) if len(p) == 2]
    rng.shuffle(nli)
    return {"pool": cards, "nli": nli}


def build(name, data):
    html = (HERE / "page_template.html").read_text(encoding="utf-8")
    start, end = html.index("const SECTIONS = ["), html.index("const TONE = {}")
    html = html[:start] + (HERE / PAGES[name]["sections"]).read_text(encoding="utf-8") + html[end:]
    title = PAGES[name]["title"]
    html = re.sub(r"<title>.*?</title>", f"<title>{title}</title>", html, count=1)
    html = re.sub(r"<h1>.*?</h1>", f"<h1>{title}</h1>", html, count=1)
    assert html.count("__DATA__") == 1
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", r"<\/"))
    out = HERE / f"review_v2_{name}.html"
    out.write_text(html, encoding="utf-8")
    print(name, {k: len(v) for k, v in data.items()}, f"{len(html) / 1024:.0f} KB")


def main():
    ru, kz = corpus("ru"), corpus("kz")
    build("main", main_data(ru, kz))
    build("blind", blind_data(ru, kz))


if __name__ == "__main__":
    main()

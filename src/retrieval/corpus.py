"""The search corpus: every chunk of all 22 codes in one language (CLAUDE.md §12, 21.10.2026).

Labor and Tax carry the questions; the other 20 codes are distractors. The article title is
prepended to the chunk text at indexing time (chunking decision, 21.10.2026).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus"
INDEX = ROOT / "data/index"


def load(lang):
    """-> (ids, texts) in a fixed order (sorted by file name, then file order)."""
    ids, texts = [], []
    for p in sorted(CORPUS.glob(f"*_{lang}.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            ids.append(r["chunk_id"])
            texts.append(f"{r['title']}\n{r['text']}" if r.get("title") else r["text"])
    assert len(ids) == len(set(ids)), f"duplicate chunk ids in {lang} corpus"
    return ids, texts

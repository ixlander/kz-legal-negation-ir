"""Model pool (CLAUDE.md §12, 21.10.2026) and how each model wants its inputs.

Documents and queries are encoded differently per model: e5 uses "passage: "/"query: ",
instruction-tuned models take an instruction on the query side only, Gemma/Qwen ship
named prompts in their sentence-transformers config. Everything model-specific lives here.
"""
from sentence_transformers import SentenceTransformer

TASK = "Given a legal question, retrieve the article of the law that answers it"

EMBEDDERS = {
    "e5-base": {"id": "intfloat/multilingual-e5-base", "doc": "passage: ", "query": "query: "},
    "e5-large-instruct": {"id": "intfloat/multilingual-e5-large-instruct", "doc": "",
                          "query": f"Instruct: {TASK}\nQuery: "},
    "bge-m3": {"id": "BAAI/bge-m3", "doc": "", "query": ""},
    "labse": {"id": "sentence-transformers/LaBSE", "doc": "", "query": ""},
    "qwen3-emb-0.6b": {"id": "Qwen/Qwen3-Embedding-0.6B", "doc": "", "query": f"Instruct: {TASK}\nQuery:"},
    # Gemma embedding models ship prompts in their ST config ("document" / "query")
    "embeddinggemma-2": {"id": "google/embeddinggemma-2", "doc_prompt": "document", "query_prompt": "query"},
}

# Tokenizers for D2 only (one per vocabulary family; several pool models share XLM-R's).
TOKENIZERS = {
    "xlm-r (e5, bge-m3, gte, arctic, jina-v3)": "FacebookAI/xlm-roberta-base",
    "labse": "sentence-transformers/LaBSE",
    "qwen3 byte-BPE": "Qwen/Qwen3-Embedding-0.6B",
    "gemma BPE": "google/embeddinggemma-2",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "mdeberta-v3": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "rubert": "DeepPavlov/rubert-base-cased",
    "user-bge-m3 (ru-pruned)": "deepvk/USER-bge-m3",
    "kaz-roberta": "kz-transformers/kaz-roberta-conversational",
    "kazbert": "Eraly-ml/KazBERT",
    "hplt-kk": "HPLT/hplt_bert_base_kk",
    "kk-small bpe": "Abzalbek89/kk-bert-small-bpe",
    "kk-small morph-bpe": "Abzalbek89/kk-bert-small-morph-bpe",
    "kk-small unigram": "Abzalbek89/kk-bert-small-unigram",
}


def load(key, device=None):
    return SentenceTransformer(EMBEDDERS[key]["id"], device=device, trust_remote_code=False)


def _encode(model, key, texts, side, batch_size=16):
    cfg = EMBEDDERS[key]
    kw = dict(normalize_embeddings=True, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    prompt_name = cfg.get(f"{side}_prompt")
    if prompt_name and prompt_name in (getattr(model, "prompts", None) or {}):
        return model.encode(texts, prompt_name=prompt_name, **kw)
    return model.encode([cfg.get(side, "") + t for t in texts], **kw)


def encode_docs(model, key, texts, **kw):
    return _encode(model, key, texts, "doc", **kw)


def encode_queries(model, key, texts, **kw):
    return _encode(model, key, texts, "query", **kw)

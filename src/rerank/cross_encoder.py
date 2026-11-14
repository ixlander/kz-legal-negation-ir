"""Off-the-shelf multilingual cross-encoder rerankers (B3′)."""
from sentence_transformers import CrossEncoder

RERANKERS = {
    "jina-v2": {"id": "jinaai/jina-reranker-v2-base-multilingual", "remote_code": True},
    "bge-reranker-v2-m3": {"id": "BAAI/bge-reranker-v2-m3", "remote_code": False},
}
MAX_LEN = 512


def load(key):
    cfg = RERANKERS[key]
    model = CrossEncoder(cfg["id"], max_length=MAX_LEN, trust_remote_code=cfg["remote_code"])
    if model.device.type == "cuda":
        model.model.half()
    return model


def scores(model, query, docs, batch_size=16):
    return model.predict([(query, d) for d in docs], batch_size=batch_size, show_progress_bar=False)

"""Off-the-shelf multilingual cross-encoder rerankers (B3′)."""
import torch
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlmr
from sentence_transformers import CrossEncoder


def _create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    return ((torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask).long() + padding_idx


# jina-reranker-v2 remote code imports this helper, which transformers 5 no longer exports
if not hasattr(xlmr, "create_position_ids_from_input_ids"):
    xlmr.create_position_ids_from_input_ids = _create_position_ids_from_input_ids

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

"""Fine-tune XLM-R as a 3-class NLI model for the B3 reranker.

Variants (training data):
    a   XNLI ru + XNLI tr + synthetic ru/kz            main model
    b   a without XNLI tr                              H5: does Turkish help Kazakh?
    c   a without the entailment counter-examples      H6: do «не менее» / synonymous-negation pairs
        (synthetic range and synonym_neg entailments)      reduce false contradictions?
    d   a + SNLI-kk (machine translated)               extra H5 variant
Model selection: macro F1 on Kardeş-NLU kk dev after every epoch; the best epoch is kept.

Usage:
    python -m src.rerank.nli_train --variant a
    python -m src.rerank.nli_train --variant b --epochs 2
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
NLI, CKPT = ROOT / "data/nli", ROOT / "checkpoints"
BASE = "FacebookAI/xlm-roberta-base"
LABELS = ["entailment", "neutral", "contradiction"]
SEED, MAX_LEN = 42, 128
COUNTER_EXAMPLES = {"range", "synonym_neg"}


def jl(name):
    return [json.loads(l) for l in open(NLI / f"{name}.jsonl", encoding="utf-8")]


def training_rows(variant):
    rows = jl("xnli_ru") + jl("synthetic_ru") + jl("synthetic_kz")
    if variant != "b":
        rows += jl("xnli_tr")
    if variant == "c":
        rows = [r for r in rows if not (r.get("kind") in COUNTER_EXAMPLES and r["label"] == 0)]
    if variant == "d":
        rows += jl("snli_kk")
    return rows


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LengthGroupedBatches(Sampler):
    """Batches of similar-length pairs (less padding), in random order; the rows inside a pool of 50
    batches are sorted by length, so the order still changes every epoch."""

    def __init__(self, lengths, batch_size, seed=SEED, pool=50):
        self.lengths, self.batch_size, self.pool, self.epoch, self.seed = lengths, batch_size, pool, 0, seed

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        size = self.batch_size * self.pool
        out = []
        for i in range(0, len(idx), size):
            chunk = sorted(idx[i:i + size], key=self.lengths.__getitem__)
            out += [chunk[j:j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        rng.shuffle(out)
        return iter(out)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def batches(rows, tok, batch_size, shuffle):
    def collate(b):
        enc = tok([r["premise"] for r in b], [r["hypothesis"] for r in b], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([r["label"] for r in b])
        return enc
    if shuffle:
        lengths = [len(r["premise"]) + len(r["hypothesis"]) for r in rows]
        return DataLoader(rows, batch_sampler=LengthGroupedBatches(lengths, batch_size), collate_fn=collate)
    return DataLoader(rows, batch_size=batch_size, collate_fn=collate)


def save_fp16(model, tok, out):
    """fp16 weights halve both the file and the peak RAM of the copy to CPU; inference runs in fp16 anyway."""
    gc.collect()
    torch.cuda.empty_cache()
    state = {k: v.detach().to("cpu", dtype=torch.float16) for k, v in model.state_dict().items()}
    model.save_pretrained(out, state_dict=state)
    tok.save_pretrained(out)
    del state
    gc.collect()


@torch.no_grad()
def macro_f1(model, loader, device):
    model.eval()
    pred, gold = [], []
    for b in loader:
        b = b.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            pred.append(model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1).cpu())
        gold.append(b["labels"].cpu())
    model.train()
    return float(f1_score(torch.cat(gold), torch.cat(pred), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list("abcd"), required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)}).to(device)
    rows = training_rows(a.variant)
    train = batches(rows, tok, a.batch_size, shuffle=True)
    dev = batches(jl("kardes_kk_dev"), tok, 64, shuffle=False)
    print(f"variant {a.variant}: {len(rows)} training pairs, {len(train)} steps/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(train) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    out = CKPT / f"nli-{a.variant}"
    log = {"variant": a.variant, "base": BASE, "n_train": len(rows), "epochs": a.epochs, "lr": a.lr,
           "batch_size": a.batch_size, "max_len": MAX_LEN, "seed": SEED, "dev": "kardes_kk_dev", "history": []}
    best = -1.0
    model.train()
    for epoch in range(1, a.epochs + 1):
        t0, running = time.time(), 0.0
        for step, b in enumerate(train, 1):
            b = b.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            running += loss.item()
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}/{len(train)} loss {running / 500:.4f} ({time.time() - t0:.0f}s)")
                running = 0.0
        f1 = macro_f1(model, dev, device)
        log["history"].append({"epoch": epoch, "dev_macro_f1": f1, "seconds": round(time.time() - t0)})
        print(f"epoch {epoch}: kardes kk dev macro F1 {f1:.4f}")
        if f1 > best:
            best = f1
            save_fp16(model, tok, out)
    log["best_dev_macro_f1"] = best
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    (ROOT / f"results/nli_train_{a.variant}.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"best dev macro F1 {best:.4f} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

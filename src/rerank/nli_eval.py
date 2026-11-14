"""Evaluate an NLI classifier on the validation sets.

    kardes-kk     Kardeş-NLU Kazakh XNLI test (human post-edited), 3 classes, report half
    rcb-ru        Russian CommitmentBank val, 3 classes; also its negated-premise slice
    lidirus-neg   LiDiRus diagnostics with logic = Negation / Double negation; binary
                  (entailment vs not), a 3-class prediction counts as not-entailment unless it is entailment
    synthetic-*   held-out split of our rule-based legal pairs, reported per rewrite kind
For every set: accuracy, macro F1, contradiction recall. Range-negation Error (H6) is the share of
range pairs labelled entailment that the model calls contradiction.

Usage:
    python -m src.rerank.nli_eval --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    python -m src.rerank.nli_eval --model checkpoints/nli-a
"""
import argparse
import datetime as dt
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
NLI, EXT = ROOT / "data/nli", ROOT / "data/external"
LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}
MAX_LEN, BATCH = 256, 32


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def zipped_jsonl(archive, member):
    with zipfile.ZipFile(archive) as z:
        return [json.loads(l) for l in z.read(member).decode("utf-8").splitlines() if l.strip()]


def eval_sets():
    # odd-indexed half only: the even half selects epochs during fine-tuning
    sets = {"kardes-kk-report": jl(NLI / "kardes_kk_test.jsonl")[1::2]}
    rcb = [{"premise": r["premise"], "hypothesis": r["hypothesis"], "label": LABELS[r["label"]],
            "kind": r["negation"]} for r in zipped_jsonl(EXT / "russian_super_glue/data/RCB.zip", "RCB/val.jsonl")]
    sets["rcb-ru"] = rcb
    sets["rcb-ru-negated"] = [r for r in rcb if r["kind"] != "no_negation"]
    sets["lidirus-neg"] = [{"premise": r["sentence1"], "hypothesis": r["sentence2"],
                            "label": 0 if r["label"] == "entailment" else 2, "binary": True}
                           for r in zipped_jsonl(EXT / "russian_super_glue/data/LiDiRus.zip", "LiDiRus/LiDiRus.jsonl")
                           if re.search(r"negation", r.get("logic", ""), re.I)]
    for lang in ("ru", "kz"):
        sets[f"synthetic-{lang}-test"] = [r for r in jl(NLI / f"synthetic_{lang}.jsonl") if r.get("split") == "test"]
    manual = NLI / "val_manual.jsonl"
    if manual.exists():
        rows = jl(manual)
        sets["manual"] = [r for r in rows if r.get("status") == "ok"]
        for lang in ("ru", "kz"):
            # unchecked drafts: indicative only, never reported as a result
            sets[f"manual-draft-{lang}"] = [{**r, "kind": "range" if r["range"] else "other"}
                                            for r in rows if r["lang"] == lang]
    return {k: v for k, v in sets.items() if v}


class Classifier:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()
        # model label order -> ours (0 entailment, 1 neutral, 2 contradiction)
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.order = [next(i for i, l in id2label.items() if l.startswith(name_)) for name_ in LABELS]

    @torch.no_grad()
    def probs(self, premises, hypotheses):
        out = []
        for i in range(0, len(premises), BATCH):
            enc = self.tok(premises[i:i + BATCH], hypotheses[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            p = self.model(**enc).logits.float().softmax(-1)[:, self.order]
            out.append(p.cpu().numpy())
        return np.concatenate(out)


def scores(rows, probs):
    y = np.array([r["label"] for r in rows])
    pred = probs.argmax(1)
    if rows and rows[0].get("binary"):
        pred = np.where(pred == 0, 0, 2)
    out = {"n": len(rows), "accuracy": float((pred == y).mean()),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "contradiction_recall": float((pred[y == 2] == 2).mean()) if (y == 2).any() else None}
    kinds = sorted({r["kind"] for r in rows if "kind" in r})
    if kinds:
        out["by_kind"] = {k: {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
                          for k in kinds for m in [np.array([r.get("kind") == k for r in rows])]}
    rng = np.array([r.get("kind") == "range" and r["label"] == 0 for r in rows])
    if rng.any():
        out["range_negation_error"] = float((pred[rng] == 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None, help="name in the result file; defaults to the model name")
    a = ap.parse_args()
    clf = Classifier(a.model)
    res = {"model": a.model, "date": dt.date.today().isoformat(), "sets": {}}
    for name, rows in eval_sets().items():
        res["sets"][name] = scores(rows, clf.probs([r["premise"] for r in rows], [r["hypothesis"] for r in rows]))
        s = res["sets"][name]
        print(f"{name:16s} n={s['n']:5d} acc {s['accuracy']:.3f} F1 {s['macro_f1']:.3f}"
              + (f" range-err {s['range_negation_error']:.3f}" if "range_negation_error" in s else ""))
    tag = a.tag or Path(a.model).name
    path = ROOT / f"results/nli_eval_{tag}_{res['date']}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

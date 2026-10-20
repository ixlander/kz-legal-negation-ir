"""D2 Affix Isolation on many tokenizers (src/models.py TOKENIZERS).

Forms: the verified ones (status ok / ok_auto) from data/diagnostics/d2_kz_neg_forms.tsv.
Besides isolation, reports how often a form contains an unknown token (vocabularies pruned
to Russian may have lost the Kazakh letters ә ғ қ ң ө ұ ү і һ).

Usage:
    python -m src.diagnostics.d2_pool
"""
import json
from datetime import date

from transformers import AutoTokenizer

from src.diagnostics.d2_affix import ROOT, analyse_kz, analyse_ru, load_forms
from src.models import TOKENIZERS

KZ_LETTERS = "әғқңөұүіһ"


def main():
    rows = load_forms(ROOT / "data/diagnostics/d2_kz_neg_forms.tsv", include_unchecked=False)
    out = {"diagnostic": "D2_affix_isolation_pool", "n_forms": len(rows), "date": date.today().isoformat(),
           "tokenizers": {}}
    print(f"{'tokenizer':<42} {'iso':>6} {'iso_w':>6} {'start':>6} {'buried':>6} {'unk':>5} {'kzlet':>5} {'ru-ne-':>6}")
    for name, repo in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            out["tokenizers"][name] = {"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{name:<42} LOAD FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        try:
            kz, per = analyse_kz(tok, rows)
            ru, _, _ = analyse_ru(tok)
        except Exception as e:  # e.g. no offset mapping (slow tokenizer)
            out["tokenizers"][name] = {"repo": repo, "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{name:<42} FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        unk = tok.unk_token
        n_unk = sum(any(t == unk for t in p["tokens"]) for p in per) if unk else 0
        kz_letters_ok = sum(tok.convert_tokens_to_ids(tok.tokenize(c))[0] != tok.unk_token_id for c in KZ_LETTERS)
        out["tokenizers"][name] = {"repo": repo, "vocab_size": len(tok), "kz": kz, "ru_control": ru,
                                   "forms_with_unk": n_unk, "kz_letters_known": f"{kz_letters_ok}/{len(KZ_LETTERS)}",
                                   "examples": [{"split": p["split"], "tokens": p["tokens"]} for p in per[:12]]}
        print(f"{name:<42} {kz['isolation_rate']:>6.1%} {kz['isolation_rate_freq_weighted']:>6.1%} "
              f"{kz['patterns']['starts_token']:>6.1%} {kz['patterns']['buried']:>6.1%} {n_unk:>5} "
              f"{kz_letters_ok:>3}/{len(KZ_LETTERS)} {ru['fused_prefix_boundary_rate']:>6.0%}")
    (ROOT / "results/d2_pool.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Download the external datasets approved on 21.10.2026 (CLAUDE.md §12) into data/external/.

Not versioned (gated / third-party licences). Re-run is idempotent.

Usage:
    python -m src.data.fetch_external
"""
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data/external"

HF = {
    "kazqad-retrieval": ("issai/kazqad-retrieval", "dataset"),          # gated, test = native questions
    "ru-law-retrieval": ("alekseevpavel04/ru-law-retrieval", "dataset"),
    "nevir": ("orionweller/NevIR", "dataset"),
    "russian_super_glue": ("RussianNLP/russian_super_glue", "dataset"),  # RCB, LiDiRus, TERRa
    "snli-kazakh": ("sultanbi/snli-kazakh", "dataset"),
    "xnli": ("facebook/xnli", "dataset"),
}
GIT = {"kardes-nlu": "https://github.com/lksenel/Kardes-NLU.git"}


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    for name, (repo, kind) in HF.items():
        kw = {}
        if name == "xnli":  # only the languages we use
            kw["allow_patterns"] = ["ru/*", "tr/*", "README.md"]
        if name == "russian_super_glue":
            kw["allow_patterns"] = ["*RCB*", "*LiDiRus*", "*TERRa*", "*.md", "*.py", "*.json"]
        try:
            p = snapshot_download(repo, repo_type=kind, local_dir=EXT / name, **kw)
            print("OK ", name, p)
        except Exception as e:
            print("FAIL", name, type(e).__name__, str(e)[:200])
    for name, url in GIT.items():
        dst = EXT / name
        if not dst.exists():
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dst)], capture_output=True, text=True)
            print("OK " if r.returncode == 0 else "FAIL", name, r.stderr[:200])
        else:
            print("OK ", name, "(exists)")


if __name__ == "__main__":
    main()

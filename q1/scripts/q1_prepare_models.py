"""Download pinned, non-bundled BERT and NLTK files; warm WhisperX alignment cache."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile


BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
BERT_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
PUNKT_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
ALIGN_SHA256 = "488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-align-cache", action="store_true", help="WhisperX can download this at first extraction")
    parser.add_argument("--check-only", action="store_true", help="Verify local BERT and NLTK files without downloading")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bert = root / "models/bert-base-uncased"
    if not args.check_only:
        bert.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="google-bert/bert-base-uncased", revision=BERT_REVISION,
                          local_dir=str(bert), allow_patterns=["config.json", "model.safetensors",
                                                             "tokenizer.json", "tokenizer_config.json", "vocab.txt"])
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        if not (bert / name).is_file():
            raise FileNotFoundError(bert / name)
    check(bert / "model.safetensors", BERT_SHA256)
    nltk_base = root / "models/nltk_data/tokenizers"
    if not args.check_only:
        nltk_base.mkdir(parents=True, exist_ok=True)
    archive = nltk_base / "punkt_tab.zip"
    if not archive.exists():
        if args.check_only:
            raise FileNotFoundError(archive)
        url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
        urllib.request.urlretrieve(url, archive)
    check(archive, PUNKT_SHA256)
    if not args.check_only:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(nltk_base)
    if not (nltk_base / "punkt_tab/english").is_dir():
        raise RuntimeError("punkt_tab/english not found after extraction")
    if not args.skip_align_cache and not args.check_only:
        import whisperx
        whisperx.load_align_model(language_code="en", device="cpu", model_name=None)
        align = Path.home() / ".cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
        if align.is_file():
            check(align, ALIGN_SHA256)
        else:
            print("WhisperX loaded; alignment checkpoint path differs from the reference machine")
    print("Model preparation complete:", bert, nltk_base)


if __name__ == "__main__":
    main()

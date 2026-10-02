# Installation

## CPU code verification

Python 3.11 or 3.12 is sufficient for the root utilities and synthetic tests.
They use only the standard library. No installation, model download or GPU is
required. The root `.python-version` records 3.12.

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

Optional evidence re-scoring needs a separately supplied private bundle;
[DATA](DATA.md) documents its expected layout. It is not included in the checkout.

## STATE-MSA and TRACE-MSA

Use Python 3.12 and `uv`. The model environment has a locked PyTorch 2.5.1 /
NumPy 2.1.3 / Transformers 4.57.6 stack. On Linux x86_64, its lock selects the
CUDA 12.4 PyTorch index. Full training needs compatible hardware and external
assets; `--help` does not load weights.

```bash
cd sentiment_model
uv sync --locked --python 3.12
PYTHONPATH=src uv run --no-sync python -m q2.run --help
cd ../explainability
../sentiment_model/.venv/bin/python -m src.q3.run --help
```

TRACE-MSA shares the model environment. NumPy alone is enough for its arithmetic
contract tests; full checkpoint inference additionally loads PyTorch/BERT.

## Video extraction

Keep extraction in its own environment: its PyTorch 2.8 dependency differs from
the predictor's 2.5.1 stack. FFmpeg/FFprobe and OpenFace are external executables.
WhisperX alignment, openSMILE, pretrained assets and a transcript spreadsheet
are needed for actual extraction.

```bash
cd feature_extraction
uv sync --locked --python 3.12
```

Follow the [feature guide](../feature_extraction/README.md) for prerequisites.
Source-time alignment in TRACE-MSA additionally needs extraction dependencies;
prepare a suitable separate environment for that optional mode.

## External assets

The dataset is a trusted field-major `aligned_50.pkl`. Each of the three frozen
checkpoints needs its `scaler.npz`; inference uses the matching local
`bert-base-uncased` files. The exact task/scaler/BERT hashes are in
[final_model.json](../sentiment_model/config/final_model.json). Downloads for
these task checkpoints are not currently published. [DATA](DATA.md) documents
external assets, and [REPRODUCE](REPRODUCE.md) lists commands.

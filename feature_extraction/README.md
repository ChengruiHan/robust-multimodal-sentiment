# Feature extraction

This module extracts traceable word-level features from video and transcript. It preserves the original word order, records alignment quality, and marks audio/visual positions unavailable when a trustworthy alignment cannot be established.

The output is a per-sample NPZ/JSON pair with text `[L, 768]`, audio `[L, 50]`, and visual `[L, 40]` features, where $L$ is the number of words in the sample. It is **not directly compatible** with the 74/35-dimensional audio/visual input used by [STATE-MSA](../sentiment_model/). See [Data contracts](../docs/data-contracts.md).

## Requirements

- Python 3.12 and `uv` for the locked Python environment.
- FFmpeg and FFprobe on `PATH`.
- OpenFace 2.2.0 `FeatureExtraction`, with its `model/` and `AU_predictors/` beside the executable.
- Local BERT, NLTK, and WhisperX alignment assets prepared by the model setup command.
- Authorized source videos and a `label-100.xlsx` spreadsheet in the expected input layout.

## Workflow

From this directory:

```bash
uv sync --locked --python 3.12
uv run --no-sync python -m scripts.q1_prepare_models
uv run --no-sync python -m scripts.q1_package_setup \
  --source-dir /absolute/path/to/source-videos \
  --openface /absolute/path/to/OpenFace/FeatureExtraction \
  --output-dir /absolute/path/to/new-output
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage audit
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage preflight
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage run
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage validate
```

The setup command requires the spreadsheet and video folders expected by the original data adapter. The checked-in review CSV files contain only column headers; add sample-specific review decisions locally when needed. `configs/q1.yaml`, downloaded models, generated features, and outputs are ignored by Git.

The source adapter and names in `scripts/` retain their original data contract. They are useful for reproducing this extraction study; adapting to another dataset requires changing the input adapter and validating the resulting alignment.

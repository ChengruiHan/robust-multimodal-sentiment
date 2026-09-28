# Data contracts

The repository contains two feature schemas. They describe distinct experimental paths.

## STATE-MSA and TRACE-MSA aligned data

The predictor reads a trusted `aligned_50.pkl` file containing `train`, `valid`, and `test` dictionaries. Each split is **field-major**: `data[split][field][sample_index]`. Required fields are `id`, `text_bert`, `audio`, `vision`, `classification_labels`, and `regression_labels`.

| Field | Per-sample shape | Meaning |
| --- | --- | --- |
| `text_bert` | `[3, 50]` | BERT token IDs, attention mask, token type IDs; the model encodes these to text representations |
| `audio` | `[50, 74]` | Aligned acoustic features |
| `vision` | `[50, 35]` | Aligned visual features |
| `classification_labels` | scalar | Three-class target: negative, neutral, positive |
| `regression_labels` | scalar | Intensity target in approximately `[-3, 3]` |

The loader constructs content and availability masks from the aligned fields. A zero-valued audio or visual position is treated as unavailable by this implementation; that convention is part of this dataset contract. The audio/visual scaler is fitted only on observed positions in the training split. See [`sentiment_model/src/q2/data.py`](../sentiment_model/src/q2/data.py) for validation and transformation details.

TRACE-MSA additionally accepts aligned sample files with `id`, `raw_text`, `text_bert`, `audio`, and `vision`. Optional video files support source-time mapping. The included inference entry point expects the original experiment's numbered file layout; see [`explainability/src/q3/run.py`](../explainability/src/q3/run.py).

## Video feature extraction output

The extraction module starts from a transcript spreadsheet and source videos. Its per-sample NPZ preserves the actual number of words $L$:

| Field | Shape | Meaning |
| --- | --- | --- |
| `text_features` | `[L, 768]` | BERT word representation |
| `audio_features` | `[L, 50]` | Aggregated acoustic descriptors |
| `visual_features` | `[L, 40]` | Aggregated facial-behavior descriptors |

The NPZ and sidecar JSON also retain alignment and source-trace metadata. This output is **not input-compatible** with the 74/35-dimensional aligned data above. Resampling positions alone would not make the feature definitions equivalent. See [`feature_extraction/`](../feature_extraction/) for the extraction workflow.

## Asset handling

Dataset files, videos, pretrained weights, model checkpoints, fitted scalers, and generated outputs are not stored in this repository. Obtain them only from authorized sources. Pickle and PyTorch checkpoint deserialization can execute unsafe content, so use trusted files only.

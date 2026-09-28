# STATE-MSA sentiment model

STATE-MSA predicts a three-class sentiment label and continuous intensity from an aligned text/audio/visual sample. It uses observed-position masks, a multimodal backbone, and a conditional intensity head. The recorded configuration ensembles three seeds.

## Code map

| Path | Purpose |
| --- | --- |
| `src/q2/data.py` | Aligned-data contract, observed masks, train-fitted audio/visual scaler |
| `src/q2/missing.py` | Controlled missing-modality masks and evaluation grid |
| `src/q2/model.py` | Multimodal predictor |
| `src/q2/conditional_reg.py` | Conditional regression head |
| `src/q2/run.py` | Audit, training, validation, and original sample inference CLI |
| `scripts/` | Reproduction scripts for the recorded three-seed setup |
| `config/final_model.json` | Recorded model configuration and asset hashes |

## Set up

Use Python 3.12 and `uv` from this directory:

```bash
uv sync --locked
PYTHONPATH=src uv run --no-sync python -m q2.run --help
```

The model expects a trusted `aligned_50.pkl` file with 50 aligned positions and audio/visual dimensions 74/35, plus compatible local `bert-base-uncased` weights. See [Data contracts](../docs/data-contracts.md). The video feature extraction module has a different schema and cannot supply this input directly.

## Train the recorded configuration

Provide absolute paths through environment variables and choose a new output directory:

```bash
DATA=/absolute/path/to/aligned_50.pkl \
BERT_DIR=/absolute/path/to/bert-base-uncased \
OUT=/absolute/path/to/new-training-output \
bash scripts/train_final.sh
```

This trains the three recorded seeds and their conditional intensity heads, then computes the validation condition grid. It requires the original compatible data and substantial compute. The repository does not include a pretrained release checkpoint.

The `evaluate_frozen.sh` and `predict_attachment3.sh` scripts support the original experiment's file layout and require matching checkpoints and fitted scalers. Their names are retained for reproducibility. Validation figures in the root README are [recorded results](../docs/evaluation.md), not rerun results from this public checkout.

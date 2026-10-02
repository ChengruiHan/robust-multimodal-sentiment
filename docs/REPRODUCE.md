# Reproduce the released workflow

Run each block from the directory named in its heading. Choose fresh output
paths: the evidence re-scorer and model shell runners reject existing outputs.

## CPU code verification — repository root

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

Tests check arithmetic and input validation using synthetic fixtures. The
public release includes no original prediction or explanation records.

## Optional private evidence re-scoring — repository root

```bash
python scripts/verify_release.py --root /absolute/path/to/private-evidence
python scripts/rescore.py --root /absolute/path/to/private-evidence \
  --output outputs/local_rescore.json
```

Prepare the private bundle using the schema in [DATA](DATA.md). This path checks
hashes, condition metrics, record consistency and fidelity subtraction/mean.
It runs no model inference, deletion experiments or bootstrap interval analysis.
It is specific to the original validation layout and does not upload records.

## Contract tests

These synthetic tests do not need original data or pretrained weights. Install
NumPy and openpyxl for extraction tests; the model tests also need PyTorch.
Run them in the appropriate environments from [INSTALL](INSTALL.md).

```bash
(cd feature_extraction && .venv/bin/python -m unittest discover -s tests -v)
(cd sentiment_model && PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v)
(cd explainability && ../sentiment_model/.venv/bin/python -m unittest discover -s tests -v)
```

GitHub Actions runs these suites with a small CPU test environment. These
checks establish software contracts rather than reproduced training accuracy.

## Train STATE-MSA — sentiment_model

```bash
DATA=/absolute/path/to/aligned_50.pkl \
BERT_DIR=/absolute/path/to/bert-base-uncased \
OUT=/absolute/path/to/fresh-training-output \
bash scripts/train_final.sh
```

The runner audits data, trains M4 with seeds 3407/42/2026, fits each two-regime
conditional head, evaluates each 55-condition grid and averages seed outputs.
Training and selection settings are in the runner and
[final_model.json](../sentiment_model/config/final_model.json).
Newly trained weights will generally differ from the archived hashes.

## Evaluate archived weights — sentiment_model

```bash
DATA=/absolute/path/to/aligned_50.pkl \
BERT_DIR=/absolute/path/to/bert-base-uncased \
WEIGHTS_DIR=/absolute/path/to/frozen-weights \
SPLIT=valid \
OUT=/absolute/path/to/fresh-frozen-evaluation \
bash scripts/evaluate_frozen.sh
```

`WEIGHTS_DIR` must contain `seed_3407`, `seed_42`, `seed_2026`, each with `best.pt`
and `scaler.npz`. The runner verifies their hashes and BERT against the recorded
manifest before inference. `SPLIT=test` uses the existing test adapter. Newly
trained weights can be evaluated using the underlying `q2.run grid` command;
they should not be substituted for archived assets in a historical comparison.

For the original 30 unlabelled aligned samples:

```bash
ATTACHMENT3_DIR=/absolute/path/to/aligned-samples \
BERT_DIR=/absolute/path/to/bert-base-uncased \
WEIGHTS_DIR=/absolute/path/to/frozen-weights \
OUT=/absolute/path/to/fresh-sample-predictions \
bash scripts/predict_attachment3.sh
```

These samples have no supervised labels and yield predictions, not accuracy.

## Validate TRACE-MSA — explainability

```bash
../sentiment_model/.venv/bin/python -m src.q3.run validate \
  --data /absolute/path/to/aligned_50.pkl \
  --checkpoints /absolute/path/to/seed_3407/best.pt \
                /absolute/path/to/seed_42/best.pt \
                /absolute/path/to/seed_2026/best.pt \
  --bert /absolute/path/to/bert-base-uncased \
  --window 3 --fidelity --random-repeats 50 --seed 3407 --expect-q2 \
  --output /absolute/path/to/fresh-explanations
```

The default device is CPU; `--device cuda` selects CUDA. This mode executes
model interventions, unlike the CPU evidence re-scorer. Optional `attachment4`
mode and `--align` use the original numbered sample/video contracts; inspect
`--help` and the [explanation guide](../explainability/README.md).

## Extract video features — feature_extraction

```bash
uv run --no-sync python -m scripts.q1_prepare_models
uv run --no-sync python -m scripts.q1_package_setup \
  --source-dir /absolute/path/to/source-videos \
  --openface /absolute/path/to/OpenFace/FeatureExtraction \
  --output-dir /absolute/path/to/fresh-feature-output
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage audit
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage preflight
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage run
uv run --no-sync python -m scripts.q1_run --config configs/q1.yaml --stage validate
```

Prepare the spreadsheet/video layout and external tools as described in the
[feature guide](../feature_extraction/README.md). Review files are templates;
source-specific decisions remain local. Output is not input-compatible with
the predictor. Adaptation requires documented feature definitions and new
validation, rather than simply resizing arrays.

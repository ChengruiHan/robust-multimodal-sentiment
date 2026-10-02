# External data and assets

The public release contains code and documentation. Research datasets, media,
numerical predictions, explanation/fidelity records and private provenance
manifests are kept outside the published Git tree.

## Full experiment assets

STATE-MSA needs a compatible field-major `aligned_50.pkl` dataset and local
`bert-base-uncased` files. Frozen inference requires three checkpoints in
`seed_3407`, `seed_42`, and `seed_2026`, each with its fitted `scaler.npz`.
The existing public [model manifest](../sentiment_model/config/final_model.json)
records the expected task/scaler/BERT hashes. Checkpoint downloads are not
provided by this repository.

TRACE-MSA uses the same model assets and aligned samples. Its optional source-time
mapping additionally needs videos and extraction/alignment tools. Video feature
extraction needs a transcript spreadsheet, source videos, OpenFace and pretrained
alignment assets. [Data contracts](data-contracts.md) describes their two distinct
feature schemas; [INSTALL](INSTALL.md) describes dependencies.

## Optional private numerical evidence bundle

The standard-library tools support the original validation-record contract.
Users must supply their own private bundle and explicitly select it with
`--root`. The following paths are relative to that external directory:

| Path | Required contents |
| --- | --- |
| `INPUTS.sha256.json` | JSON object with a `files` mapping from relative paths to SHA-256 digests |
| `results/state_msa/valid_predictions.csv.gz` | Numerical grid predictions for the original 728 samples and 55 conditions |
| `results/state_msa/valid_55_conditions.csv` | One metric row per condition |
| `results/trace_msa/valid_predictions.csv` | Numerical explanation-path predictions with targets |
| `results/trace_msa/valid_fidelity.csv` | Top/random deletion AOPC records |
| `results/trace_msa/valid_metrics.json` | Recorded prediction metrics, confusion matrix and fidelity mean |

STATE-MSA prediction columns are `condition_id`, `sample_id`, `true_cls`,
`true_reg`, `pred_cls`, `pred_reg`, `prob_negative`, `prob_neutral`, and
`prob_positive`. TRACE-MSA predictions use the same columns except `condition_id`.
Fidelity columns include `sample_id`, `aopc`, `random_aopc`, and `delta_aopc`.
Sample IDs and targets must agree across the record groups. The recorded metric
columns are `condition_id`, `accuracy`, `macro_f1`, `mae`, and `pearson`.

These utilities are specific to the original validation layout; arbitrary-size
benchmarks require an adapted contract. Hash verification and numeric re-scoring
check internal consistency, not independent label correctness or model inference.
The tools do not upload their input or output.

## Local handling

`data/`, `results/`, weights, media, generated outputs, `INPUTS.sha256.json` and
`PROVENANCE.json` are ignored by Git. New reports default to ignored `outputs/`.
Use trusted Pickle/PyTorch artifacts from authorized sources. The public quick
start uses synthetic fixtures and does not require private data.

See [REPRODUCE](REPRODUCE.md) for commands and [evaluation](evaluation.md) for
interpretation limits.

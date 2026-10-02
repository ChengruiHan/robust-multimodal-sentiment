# TRACE-MSA explainability

TRACE-MSA analyzes a frozen three-seed STATE-MSA predictor. It evaluates all eight modality subsets for exact three-player Shapley values, removes short aligned-position windows for local sensitivity, and links selected evidence to source text and approximate media time when possible.

## Code map

| Path | Purpose |
| --- | --- |
| `src/q3/core.py` | Frozen predictor loader, modality attribution, local windows |
| `src/q3/trace.py` | Transcript spans and optional source-time mapping |
| `src/q3/run.py` | Validation and aligned-sample inference CLI |
| `src/q3/report.py` | Figures and diagnostic summaries |
| `src/q3/width_sensitivity.py` | Local-window width comparison |

The loader uses [`../sentiment_model/`](../sentiment_model/) for the model and [`../feature_extraction/`](../feature_extraction/) for optional alignment helpers. It no longer keeps duplicated source copies. `--q2-root` and `--q1-root` can override the default sibling paths.

## Use with compatible assets

Run from this directory with the environment created for [`sentiment_model/`](../sentiment_model/):

```bash
../sentiment_model/.venv/bin/python -m src.q3.run --help
```

The CLI requires three compatible model checkpoints in seed order 3407, 42, 2026; each checkpoint's directory must include its fitted `scaler.npz`. It also requires the same local BERT weights used for those checkpoints. Validation mode reads the original aligned split. The sample-inference mode still uses the original numbered `attachment4` file contract. See [Data contracts](../docs/data-contracts.md) before adapting it to another collection.

For source-time mapping, `--align` additionally requires source videos, FFmpeg/FFprobe, WhisperX alignment assets, and the feature-extraction dependencies. An unresolved or automatic mapping is reported as such; a representative frame is not a frame-level causal explanation. The method and interpretation limits are in [Method](../docs/method.md).

## Reproduction and validation

See the [CPU quick start](../README.md#quick-start),
[full reproduction commands](../docs/REPRODUCE.md), and
[external asset guide](../docs/DATA.md). The public checkout includes code and
documentation; synthetic checks use no research records. Full component runs
and optional evidence re-scoring require separately supplied private assets.

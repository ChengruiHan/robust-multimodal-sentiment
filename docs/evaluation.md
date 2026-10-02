# Evaluation and interpretation

## Recorded setup

The original experiment used a three-seed M4 ensemble with a two-regime intensity
head. Its grid has 55 conditions, each using the same 728 validation samples.
One condition is complete input. The other 54 combine six affected modality
subsets (T, A, V, T+A, T+V, A+V), three partial-deletion proportions (10%, 25%,
40%), and front/middle/end contiguous locations.

| Evaluation | Accuracy | Macro-F1 | MAE |
| --- | ---: | ---: | ---: |
| Complete input | 0.6429 | 0.6279 | 0.5660 |
| Unweighted mean of 54 missing conditions | 0.6135 | 0.5981 | 0.5917 |

These values are transcribed from the original project notes. The public
repository has no per-sample prediction records or pretrained task checkpoints,
so this table is a recorded experiment rather than a turnkey benchmark result.

Macro-F1 gives equal weight to the three classes. MAE measures continuous
intensity error. Pearson is also computed by the model evaluation code and
measures linear association between target and prediction. The missing-condition
mean averages separately computed metrics, excluding clean input; it does not
pool the predictions. Masking concerns content positions and preserves at least
one content position, rather than removing entire modalities.

## TRACE-MSA evaluation

TRACE-MSA evaluates the frozen ensemble's modality coalitions and local window
interventions. Its top-versus-random deletion fidelity and window-width analysis
support inspecting model behavior. Their research records remain private.

Optional [private re-scoring](REPRODUCE.md) can check numerical prediction
metrics, confusion matrices and fidelity arithmetic when compatible external
records are provided. It does not rerun model interventions or regenerate
bootstrap confidence intervals and window-width analyses. Separately recorded
inference paths may have numerical differences; retain their identities and
avoid combining their outputs into one metric table.

## Interpretation boundaries

- The validation split was used for development and checkpoint selection;
  these figures are not untouched blind-test estimates.
- All conditions reuse the same samples and are correlated interventions.
  Their mean does not estimate naturally missing deployment performance.
- Missingness ratios count valid content positions, not seconds of video or
  percentages of an entire padded sequence.
- Shapley values and local deletions describe the model under chosen masks,
  not ground-truth human evidence or causal sentiment effects.
- Wider deletion windows remove more content. Larger effects alone do not
  establish a better explanation method.
- Automatic source-time mappings are approximate and require review.

Original unlabelled sample adapters support predictions only; supervised
metrics require labels. This release makes no new test-score or benchmark
superiority claim. See [method](method.md) and the
[model configuration](../sentiment_model/config/final_model.json).

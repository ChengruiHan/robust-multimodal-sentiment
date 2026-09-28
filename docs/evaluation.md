# Evaluation notes

## Recorded setup

The original project configuration used a three-seed STATE-MSA ensemble on an aligned multimodal dataset. It recorded a 728-sample validation split and evaluated complete input plus 54 controlled partial-modality conditions. The missingness grid combines six affected modality subsets, three proportions, and three contiguous positions. Including complete input gives 55 conditions.

| Evaluation | Accuracy | Macro-F1 | MAE |
| --- | ---: | ---: | ---: |
| Complete input | 0.6429 | 0.6279 | 0.5660 |
| Mean of 54 missing conditions | 0.6135 | 0.5981 | 0.5917 |

These values are transcribed from the original experiment notes. They were not independently recomputed while restructuring this public repository. The required dataset and trained assets are not published here, so the table is a **recorded experiment**, not a turnkey benchmark result.

## What the metrics do and do not show

- Accuracy and Macro-F1 summarize three-class validation predictions; MAE summarizes intensity error.
- The missing-condition row is an unweighted mean across the specified conditions. It is not the performance of a naturally missing deployment sample.
- The validation split was used during model development. These figures should not be presented as untouched blind-test estimates.
- TRACE-MSA's explanations use the frozen ensemble and masking interventions. Their scores describe model behavior, not ground-truth human evidence.

The model configuration is recorded in [`sentiment_model/config/final_model.json`](../sentiment_model/config/final_model.json). The code for evaluating the condition grid is in [`sentiment_model/src/q2/`](../sentiment_model/src/q2/).

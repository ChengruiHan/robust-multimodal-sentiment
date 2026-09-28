# Method notes

## Problem formulation

Let $X=(X_T,X_A,X_V)$ denote text, audio, and visual features. For each sample, an availability vector $A \in \{0,1\}^3$ identifies observed modalities. The predictor receives the observed signals together with availability information and estimates sentiment class and intensity:

$$
\hat{Y}, \hat{R} = f(X_{\mathrm{observed}}, A).
$$

This formulation is intended to prevent the model from interpreting a missing signal as a neutral observation. Padding, naturally unavailable features, and experimentally masked features should remain distinguishable in data handling and evaluation.

## Feature and alignment stage

The first stage extracts textual representations and frame-level audio/face descriptors, then aggregates them over word intervals. Alignment is treated as uncertain: low-confidence or failed alignment should be exposed through validity and quality metadata, not hidden by a plausible-looking zero vector. Source positions and timestamps are retained where available to support later inspection.

## Prediction stage

STATE-MSA operates on word-level multimodal features and explicit masks. Training and evaluation consider clean inputs alongside controlled modality-missing conditions. The final project configuration uses three random seeds; class probabilities and clipped intensity estimates are averaged across the ensemble. Intensity regression uses separate regimes conditioned on sentiment strength.

## Explanation stage

TRACE-MSA freezes the prediction ensemble. It computes exact three-player Shapley values by evaluating all modality subsets, using predicted-class log-odds as the class attribution target. Signed values preserve whether a modality supports or suppresses the selected class; normalized absolute values summarize relative contribution magnitude.

For local evidence, the method masks contiguous token windows and measures the change in model output. If the alignment metadata supports it, token positions are mapped to transcript spans and approximate audio/video time. Missing or ambiguous mappings are left unresolved rather than assigned fabricated timestamps.

## Evaluation cautions

- Controlled masking is a diagnostic, not a substitute for real-world missingness data.
- Shapley values describe behavior of the fixed model under a chosen masking baseline; they do not establish causal influence in the source recording.
- Window deletion can create inputs outside the training distribution. Its scores should be interpreted as sensitivity evidence.
- Time localization depends on automatic alignment and is approximate unless manually reviewed.
- The repository's recorded validation values were not regenerated during the showcase curation.

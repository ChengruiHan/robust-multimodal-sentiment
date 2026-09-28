# Method

## Prediction with partial observations

Let $X=(X_T,X_A,X_V)$ be aligned text, audio, and visual features. An availability tensor $A$ marks the observed positions in each modality. STATE-MSA predicts a sentiment class $Y$ and continuous intensity $R$ from the observed features and their availability:

$$
(\hat Y, \hat R) = f(X_{\mathrm{observed}}, A).
$$

The loader distinguishes padding from valid content and derives modality availability for each aligned position. Training can apply controlled masks to observed positions. Model selection and validation use separate splits; fitted audio and visual scaling parameters come from the training split.

The recorded configuration uses three random seeds of the M4 backbone, with a conditional regression head for intensity. At inference, the ensemble averages class probabilities and intensity estimates from the three models. The implementation and its command options are in [`sentiment_model/`](../sentiment_model/).

## Explanation of a frozen predictor

TRACE-MSA evaluates all eight subsets of the three modalities for a fixed predictor. For a predicted class, it computes exact three-player Shapley values on that class's log-odds. The signed value says whether a modality raises or lowers the selected score under the masking intervention; the normalized absolute values describe relative effect magnitude.

It also masks short contiguous windows in one modality at a time and measures the change in output. When source metadata and alignment are available, selected positions can be linked to transcript spans and approximate media times. Ambiguous mapping remains unresolved. The implementation is in [`explainability/`](../explainability/).

## Traceable extraction

The independent [`feature_extraction/`](../feature_extraction/) module starts from video and transcript, extracts text, acoustic, and facial descriptors, and aggregates frame observations over word intervals. It records alignment quality and availability instead of treating every generated vector as reliable evidence.

Its output dimensions differ from those accepted by STATE-MSA. Bridging the two would require documented feature adaptation and new model validation; the current repository does not claim an end-to-end path between them.

## Interpretation boundaries

- Controlled missingness is an evaluation intervention. It does not represent every deployment condition.
- Shapley values and window deletion describe the frozen model under chosen masks. They are not causal claims about a person's sentiment.
- Window deletion may produce examples outside the training distribution.
- Media timestamps from automatic alignment are approximate and need human review for precise claims.

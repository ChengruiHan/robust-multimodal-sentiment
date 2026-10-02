# Experimental history and release scope

The work originated in a mathematical modeling study with numbered tasks:
word-level extraction (Q1), prediction with incomplete observations (Q2), and
explanation/source traceability (Q3). The public components are now named
`feature_extraction`, `sentiment_model`, and `explainability`.

## Predictor development

The retained model source supports M1–M5 backbone variants and additional
classification/fusion experiments. The released mainline uses M4 with frozen
BERT, availability-aware fusion and three fixed seeds: 3407, 42 and 2026.

Intensity experiments included a linear head, a mild/strong conditional head,
and a signed negative/mild/positive expert head. The recorded final task asset manifest uses the
**mild/strong two-regime head**. The signed
variant shares classification outputs but has different regression scores;
its metrics must not be mixed into the mainline table.

The public release contains method code and reproduction documentation.
Development runs, candidate searches, test comparisons, numerical records,
optimizer states and checkpoints remain external. The validation split was
used for development and should not be described as a blind final test.

## Explanation development

TRACE-MSA explains the frozen ensemble with all eight modality coalitions and
local deletion windows. Width 3 is the recorded main explanation setting;
widths 1 and 5 provide sensitivity comparisons. Prediction, fidelity and
width-comparison records remain private;
optional CPU utilities accept separately supplied numerical records.
Source-time mapping requires optional alignment tools and media.

## Extraction development

The video path records reversible word spans, frame aggregation, quality and
review decisions. It has a different 50/40 audio/visual feature schema from
the predictor's aligned 74/35 schema. The release preserves this distinction.

## Public organization

[OptiCall](https://github.com/ChengruiHan/OptiCall) provides the presentation
reference: method, workflow, evidence, CPU validation and full reproduction.
This repository follows that reading path while retaining its established
module entry points and independently locked environments. `RAMP`, `q1`, `q2`,
`q3`, and attachment names remain internal compatibility identifiers.

[DATA](DATA.md) describes external assets and optional private bundle schemas.
[REPRODUCE](REPRODUCE.md) is the command reference for current public paths.
Private research evidence is kept outside the published Git tree and history.

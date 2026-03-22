# Repository map

This repository separates the main pipeline from the development history.

## Main code

- `src/alignment/` handles transcript to audio alignment.
- `src/prosody/` computes acoustic features.
- `src/speaker_labeling/` handles speaker assignment.
- `src/dataset_building/` creates modeling and reporting tables.
- `src/models/` trains the model variants.
- `src/stats/` runs the statistical analyses.
- `src/visualization/` builds thesis figures.

## Archive

The archive keeps scripts that were useful during the thesis but are either narrower, more exploratory, or duplicated by cleaner versions in `src/`.

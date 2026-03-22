# Gleason Thesis Repository

Cleaned repository skeleton for the Gleason thesis pipeline.

## Main folders

- `src/alignment/` — transcript/audio alignment and ASR
- `src/prosody/` — prosody extraction
- `src/speaker_labeling/` — assign speaker labels from `.cha`
- `src/dataset_building/` — build training / GLMM / trial tables
- `src/models/` — MLM and classifier training
- `src/stats/` — statistical analyses
- `src/visualization/` — plots and thesis figures
- `src/utils/` — small helpers
- `archive/` — older exploratory or one-off scripts

## Important note

A lot of the original scripts use absolute local paths from the thesis machine.  
They are preserved here as they were, so the logic stays the same.  
Before reusing them on another machine, replace those paths or turn them into command-line arguments.

## Not included

One script mentioned in the chat, `run_ablation_cls.py`, was pasted only partially, so it is **not** included here as a runnable file.

## Suggested next cleanup pass

1. replace absolute paths with CLI arguments or config files  
2. merge person-specific speaker-labeling scripts into one generic script  
3. move large outputs to `artifacts/` and keep them out of git  
4. add small test data examples

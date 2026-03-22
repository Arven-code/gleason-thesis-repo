# Gleason Thesis Project

This repository contains the codebase for my MA thesis project on caregiver prosody, child language learning, and multimodal modelling in the Gleason corpus.

The project asks a simple but demanding question: can the immediately preceding caregiver turn help predict whether the child’s next turn will be accurate or inaccurate, and does prosody add anything beyond text alone? To answer this, the pipeline combines transcript processing, speaker labelling, transcript-audio alignment, prosodic feature extraction, classifier training, ablation experiments, mixed-effects statistical analysis, and qualitative acoustic illustration.

Rather than treating caregiver speech as a flat textual input, the project models it as a local multimodal context. Each child response is analysed in relation to the caregiver turn that comes just before it, using both lexical material and a set of F0-based prosodic measures such as level, spread, slope, and voicing. The overall aim is to connect computational modelling with questions from child language acquisition and infant-directed speech research, while keeping the full workflow reproducible and inspectable.

## Main goals

* build a clean pipeline from raw corpus material to analysis-ready datasets
* align caregiver and child turns with audio time bounds
* extract prosodic features from caregiver speech
* train text-only, prosody-only, and early-fusion models
* test whether prosody contributes beyond text
* compare mother and father contexts statistically
* document qualitative IDS-like exemplars acoustically

## Repository structure

* `src/alignment/`
  transcript-audio alignment and ASR-based timing

* `src/prosody/`
  prosodic feature extraction from aligned caregiver turns

* `src/speaker_labeling/`
  scripts for assigning speaker labels from `.cha` transcripts

* `src/dataset_building/`
  construction of training tables, trials, and GLMM-ready datasets

* `src/models/`
  masked language modelling, classifier training, and ablation scripts

* `src/stats/`
  statistical analysis scripts for prosody effects and caregiver contrasts

* `src/visualization/`
  plots, acoustic figures, and thesis-oriented visual outputs

* `archive/`
  earlier exploratory scripts and older one-off tests kept for traceability

## Pipeline overview

1. raw transcript and audio material are cleaned and organised
2. speaker labels are recovered from the corpus transcripts
3. transcript rows are aligned to the audio signal
4. caregiver turns receive prosodic feature vectors
5. caregiver-child pairs are built for modelling
6. multimodal and ablation models are trained and evaluated
7. statistical analyses test broader caregiver-role effects
8. selected examples are visualised acoustically for qualitative discussion

## Outputs

The repository is designed to produce:

* aligned transcript CSV files
* prosody-enriched datasets
* GLMM input tables
* classifier predictions and evaluation files
* ablation outputs
* statistical summary tables
* figures used in the thesis

## Project orientation

This repository is not just a storage space for scripts. It is meant to keep the entire thesis workflow readable, reusable, and transparent, from preprocessing to final interpretation. The broader objective is to preserve the link between linguistic theory, interactional context, and computational modelling, rather than reducing the project to a black-box prediction task.

## Status

This is a research codebase developed for the thesis project. Some scripts were written as exploratory tools during analysis and are therefore kept separately from the main reusable pipeline. The core folders in `src/` contain the scripts that matter most for reproducing the main stages of the project.


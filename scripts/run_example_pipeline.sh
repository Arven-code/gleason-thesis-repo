#!/usr/bin/env bash
set -e

# Example only. Replace all paths before running.

python src/alignment/align_timebounds_whisper.py   --audio /absolute/path/to/audio.wav   --in-csv /absolute/path/to/input.csv   --out-csv /absolute/path/to/aligned.csv

python src/prosody/add_prosody_parselmouth.py   --audio /absolute/path/to/audio.wav   --csv /absolute/path/to/aligned.csv   --out /absolute/path/to/aligned_with_prosody.csv

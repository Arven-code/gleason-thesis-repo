# Pipeline overview

A typical run goes through these stages:

1. Convert or clean transcript material from `.cha`.
2. Add speaker labels to transcript rows.
3. Align transcript rows to audio with Whisper or related matching logic.
4. Extract prosodic features from aligned caregiver spans.
5. Build datasets for:
   - classifier training
   - ablation comparisons
   - GLMM analyses
   - MOT/FAT and IDS report tables
6. Train the selected model.
7. Run the statistical analysis and generate figures.

The folder structure reflects this order directly.

import os
import json
import librosa
import numpy as np
import re

audio_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/andy.mp3"
transcript_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/Gleason transcripts/Mother/andy.cha"

y, sr = librosa.load(audio_file, sr=None)
hop_length = 512
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print("Audio loaded.")
print("MFCC shape:", mfccs.shape)

with open(transcript_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

pattern = re.compile(r'^\*?([A-Z]+):\s*(.*?)(?:\s+(\d+_\d+))?\s*$')

aligned_data = []
dialogue_count = 0

for line in lines:
    line = line.strip()

    if not line:
        continue

    match = pattern.match(line)

    if not match:
        if line.startswith("="):
            print("Line did not match pattern:", line)
        continue

    speaker, utterance, timestamp_str = match.groups()
    dialogue_count += 1

    if timestamp_str:
        start_ms, end_ms = map(int, timestamp_str.split("_"))
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        start_frame = int((start_sec * sr) / hop_length)
        end_frame = int((end_sec * sr) / hop_length)

        mfcc_segment = mfccs[:, start_frame:end_frame].tolist()
    else:
        mfcc_segment = None

    aligned_data.append({
        "speaker": speaker,
        "utterance": utterance,
        "timestamp": timestamp_str if timestamp_str else None,
        "mfcc_segment": mfcc_segment
    })

output_file = "aligned_data.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(aligned_data, f, indent=2, ensure_ascii=False)

print(f"\nProcessed {dialogue_count} dialogue lines.")
print(f"Aligned data saved to {output_file}")

import os
import librosa
import numpy as np
import re

audio_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/andy.mp3"

y, sr = librosa.load(audio_file, sr=None)
hop_length = 512
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print("Audio loaded.")
print("MFCC shape:", mfccs.shape)

transcript_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/Gleason transcripts/Mother/andy.cha"

with open(transcript_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

sample_line = None
for line in lines:
    line = line.strip()
    if re.match(r'^\*?[A-Z]+:', line):
        sample_line = line
        break

if not sample_line:
    raise ValueError("No dialogue line found in transcript!")

print("\nSample transcript line:")
print(sample_line)

pattern = re.compile(r'^\*?([A-Z]+):\s*(.*?)(?:\s+(\d+_\d+))?\s*$')
match = pattern.match(sample_line)

if not match:
    raise ValueError("Transcript line did not match expected format!")

speaker, utterance, timestamp_str = match.groups()

print("\nParsed transcript details:")
print("  Speaker:", speaker)
print("  Utterance:", utterance)

if timestamp_str:
    print("  Timestamp:", timestamp_str)

    start_ms, end_ms = map(int, timestamp_str.split('_'))
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0

    start_frame = int((start_sec * sr) / hop_length)
    end_frame = int((end_sec * sr) / hop_length)

    mfcc_segment = mfccs[:, start_frame:end_frame]
else:
    print("  No timestamp found; alignment will be skipped for this line.")
    mfcc_segment = None

aligned_data = {
    "speaker": speaker,
    "utterance": utterance,
    "timestamp": timestamp_str if timestamp_str else None,
    "mfcc_segment": mfcc_segment.tolist() if mfcc_segment is not None else None
}

print("\nAligned Data for the Sample Dialogue:")
print(aligned_data)

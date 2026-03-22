import librosa
import numpy as np

audio_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/andy.mp3"

y, sr = librosa.load(audio_file, sr=None)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

hop_length = 512

print("Audio loaded.")
print("MFCC shape:", mfccs.shape)

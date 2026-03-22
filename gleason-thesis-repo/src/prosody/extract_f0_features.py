import librosa
import numpy as np

def extract_f0_features(path):
    y, sr = librosa.load(path, sr=22050)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    f0 = f0[~np.isnan(f0)]

    if len(f0) == 0:
        return [0] * 12

    return [
        np.mean(f0),
        np.median(f0),
        np.std(f0),
        np.percentile(f0, 75) - np.percentile(f0, 25),
        np.percentile(f0, 10),
        np.percentile(f0, 90),
        np.max(f0) - np.min(f0),
        np.std(f0) / np.mean(f0),
        f0[-1] - f0[0],
        np.diff(f0).std(),
        np.mean(voiced_flag),
        len(f0)
    ]

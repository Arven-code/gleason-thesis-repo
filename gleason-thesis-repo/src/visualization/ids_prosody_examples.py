#!/usr/bin/env python3
"""
ids_prosody_examples.py

Pulls caregiver turns (MOT/FAT) that come right before a successful child utterance
(label 2.0) for Andy (Father + Mother), and plots waveform + F0.
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

FATHER_AUDIO_PATH = (
    "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Printemps 2025/"
    "Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/Father audio/andy.mp3"
)

MOTHER_AUDIO_PATH = (
    "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Printemps 2025/"
    "Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/Audio/andy.mp3"
)

FATHER_CSV_PATH = (
    "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Printemps 2025/"
    "Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/FINAL/andy_training_echo_v2.csv"
)

MOTHER_CSV_PATH = (
    "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Printemps 2025/"
    "Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/FINAL/andy_transcript 2_pros.csv"
)

def load_transcript_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Speaker" in df.columns:
        df.rename(columns={"Speaker": "speaker"}, inplace=True)

    if "Transcript" in df.columns:
        df.rename(columns={"Transcript": "transcript"}, inplace=True)

    if "start_sec" in df.columns and "end_sec" in df.columns:
        df.rename(columns={"start_sec": "start", "end_sec": "end"}, inplace=True)
    elif "start_s" in df.columns and "end_s" in df.columns:
        df.rename(columns={"start_s": "start", "end_s": "end"}, inplace=True)
    else:
        raise ValueError("No recognisable start/end timestamp columns found.")

    return df

@dataclass
class CaregiverExample:
    cg_start: float
    cg_end: float
    cg_text: str
    child_text: str
    child_label: float

@dataclass
class SegmentStats:
    start: float
    end: float
    f0_mean: float
    f0_median: float
    f0_min: float
    f0_max: float
    f0_range: float
    n_voiced: int

def load_audio(path: str, sr: int = 16000):
    y, sr_loaded = librosa.load(path, sr=sr)
    return y, sr_loaded

def estimate_f0(y, sr, fmin: float, fmax: float):
    f0_raw, vf, vp = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    times = librosa.frames_to_time(
        np.arange(len(f0_raw)), sr=sr, hop_length=256
    )

    unvoiced = np.isnan(f0_raw)
    f0_tmp = np.copy(f0_raw)
    f0_tmp[unvoiced] = 0.0
    f0_smooth = median_filter(f0_tmp, size=5)
    f0_smooth[unvoiced] = np.nan

    return times, f0_smooth

def compute_segment_stats(times, f0, start, end) -> SegmentStats:
    mask = (times >= start) & (times <= end)
    seg = f0[mask]
    seg = seg[~np.isnan(seg)]
    seg = seg[seg > 0]

    if len(seg) == 0:
        return SegmentStats(start, end, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    return SegmentStats(
        start=start,
        end=end,
        f0_mean=float(np.mean(seg)),
        f0_median=float(np.median(seg)),
        f0_min=float(np.min(seg)),
        f0_max=float(np.max(seg)),
        f0_range=float(np.max(seg) - np.min(seg)),
        n_voiced=len(seg),
    )

def plot_waveform_and_f0(y, sr, times, f0, seg: Tuple[float, float], title: str):
    s, e = seg
    t = np.linspace(0, len(y) / sr, len(y))

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    fig, (ax_wav, ax_f0) = plt.subplots(
        2, 1, figsize=(8, 4), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    ax_wav.plot(t, y, linewidth=0.7)
    ax_wav.axvspan(s, e, color="tab:blue", alpha=0.3)
    ax_wav.set_ylabel("Amplitude")

    ax_f0.plot(times, f0, color="tab:blue", linewidth=1.0)
    ax_f0.axvspan(s, e, color="tab:blue", alpha=0.3)
    ax_f0.set_ylabel("F0 (Hz)")
    ax_f0.set_xlabel("Time (s)")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    plt.show()

def find_caregiver_success_examples(
    df: pd.DataFrame,
    caregiver_code: Literal["MOT", "FAT"],
) -> List[CaregiverExample]:
    examples: List[CaregiverExample] = []

    df = df.sort_values("start").reset_index(drop=True)

    n = len(df)
    for i in range(n):
        cg = df.iloc[i]
        if cg["speaker"] != caregiver_code:
            continue

        for j in range(i + 1, n):
            nxt = df.iloc[j]
            if nxt["speaker"] in ["CHI", "CHI2"]:
                if nxt.get("label", np.nan) in [2, 2.0]:
                    examples.append(
                        CaregiverExample(
                            cg_start=float(cg["start"]),
                            cg_end=float(cg["end"]),
                            cg_text=str(cg.get("transcript", "")),
                            child_text=str(nxt.get("transcript", "")),
                            child_label=float(nxt.get("label", np.nan)),
                        )
                    )
                break

    return examples

def run_example(
    audio_path: str,
    csv_path: str,
    caregiver: Literal["MOT", "FAT"],
    fmin: float,
    fmax: float,
):
    print(f"\n=== {caregiver} example ===")
    df = load_transcript_csv(csv_path)

    examples = find_caregiver_success_examples(df, caregiver)
    if not examples:
        print("No caregiver→next-child(success) pairs found.")
        return

    ex = examples[0]

    print(f"Caregiver turn: {ex.cg_start:.2f}–{ex.cg_end:.2f}s")
    print(f"Text: {ex.cg_text}")
    print(f"→ Child (success): {ex.child_text}\n")

    y, sr = load_audio(audio_path)
    times, f0 = estimate_f0(y, sr, fmin=fmin, fmax=fmax)
    stats = compute_segment_stats(times, f0, ex.cg_start, ex.cg_end)

    print("F0 stats (voiced frames only):")
    print(f"  mean   = {stats.f0_mean:.1f} Hz")
    print(f"  median = {stats.f0_median:.1f} Hz")
    print(f"  min    = {stats.f0_min:.1f} Hz")
    print(f"  max    = {stats.f0_max:.1f} Hz")
    print(f"  range  = {stats.f0_range:.1f} Hz")
    print(f"  voiced frames = {stats.n_voiced}")

    title = f"{caregiver} – caregiver turn before successful child utterance"
    plot_waveform_and_f0(y, sr, times, f0, (ex.cg_start, ex.cg_end), title)

if __name__ == "__main__":
    run_example(
        audio_path=FATHER_AUDIO_PATH,
        csv_path=FATHER_CSV_PATH,
        caregiver="FAT",
        fmin=60.0,
        fmax=350.0,
    )

    run_example(
        audio_path=MOTHER_AUDIO_PATH,
        csv_path=MOTHER_CSV_PATH,
        caregiver="MOT",
        fmin=120.0,
        fmax=500.0,
    )

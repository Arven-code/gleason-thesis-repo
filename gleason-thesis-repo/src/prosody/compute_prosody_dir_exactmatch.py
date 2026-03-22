#!/usr/bin/env python3
# compute_prosody_dir_exactmatch.py
# Batch prosody extraction for a whole directory of aligned CSVs.
# It matches each CSV to audio by exact basename first, then by stripped stem.

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_time_columns(df):
    start_candidates = {"start_sec", "start_s", "start", "onset", "begin"}
    end_candidates = {"end_sec", "end_s", "end", "offset", "finish"}

    start_col = next((c for c in df.columns if c.lower() in start_candidates), None)
    end_col = next((c for c in df.columns if c.lower() in end_candidates), None)
    return start_col, end_col


def build_audio_candidates(csv_stem: str):
    candidates = [csv_stem]

    suffixes = [
        "_with_speaker_training_echo_v2",
        "_training_echo_v2",
        "_with_speaker",
        "_training_echo",
    ]
    for suf in suffixes:
        if csv_stem.endswith(suf):
            candidates.append(csv_stem[: -len(suf)])

    first_token = csv_stem.split("_")[0]
    if first_token not in candidates:
        candidates.append(first_token)

    return candidates


def resolve_audio(audio_dir: str, csv_stem: str):
    candidates = build_audio_candidates(csv_stem)

    for base in candidates:
        for ext in (".wav", ".mp3"):
            cand = os.path.join(audio_dir, base + ext)
            if os.path.exists(cand):
                return cand

    return None


def compute_f0_stats(seg, sr):
    import librosa

    if len(seg) < 160:
        return dict(
            f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
            f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
            f0_slope=0.0, dF0_iqr=0.0, voicing_rate=0.0, frames=0
        )

    f0, _, _ = librosa.pyin(
        seg,
        fmin=60.0,
        fmax=700.0,
        sr=sr,
        frame_length=1024,
        hop_length=160,
    )

    frames = len(f0)
    if frames == 0:
        return dict(
            f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
            f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
            f0_slope=0.0, dF0_iqr=0.0, voicing_rate=0.0, frames=0
        )

    voiced_mask = np.isfinite(f0)
    voicing_rate = float(np.mean(voiced_mask))

    if np.sum(voiced_mask) == 0:
        return dict(
            f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
            f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
            f0_slope=0.0, dF0_iqr=0.0, voicing_rate=voicing_rate, frames=int(frames)
        )

    f = f0[voiced_mask].astype(float)
    f0_mean = float(np.mean(f))
    f0_median = float(np.median(f))
    f0_std = float(np.std(f, ddof=0))
    q25, q75 = np.percentile(f, [25, 75])
    f0_iqr = float(q75 - q25)
    p10, p90 = np.percentile(f, [10, 90])
    f0_range = float(np.max(f) - np.min(f))
    f0_cv = float(f0_std / f0_mean) if f0_mean > 1e-6 else 0.0

    idx = np.nonzero(voiced_mask)[0]
    t = idx * (160 / sr)
    f0_slope = float(np.polyfit(t, f, 1)[0]) if (len(f) >= 2 and np.std(t) > 0) else 0.0

    if len(f) >= 3:
        dfreq = np.diff(f)
        d_q25, d_q75 = np.percentile(dfreq, [25, 75])
        dF0_iqr = float(d_q75 - d_q25)
    else:
        dF0_iqr = 0.0

    return dict(
        f0_mean=f0_mean,
        f0_median=f0_median,
        f0_std=f0_std,
        f0_iqr=f0_iqr,
        f0_p10=float(p10),
        f0_p90=float(p90),
        f0_range=f0_range,
        f0_cv=f0_cv,
        f0_slope=f0_slope,
        dF0_iqr=dF0_iqr,
        voicing_rate=voicing_rate,
        frames=int(frames),
    )


def process_one(csv_path: str, audio_path: str, out_path: str):
    import librosa

    df = pd.read_csv(csv_path)
    start_col, end_col = find_time_columns(df)
    if start_col is None or end_col is None:
        print(f"[SKIP] {os.path.basename(csv_path)} -> no usable time columns")
        return

    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    rows = []
    bad = 0
    dur = len(y) / sr + 0.05

    for _, row in df.iterrows():
        try:
            st = float(row[start_col])
            en = float(row[end_col])
        except Exception:
            st, en = np.nan, np.nan

        if not (np.isfinite(st) and np.isfinite(en) and en > st and 0 <= st <= dur and 0 < en <= dur):
            rows.append(dict(
                f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
                f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
                f0_slope=0.0, dF0_iqr=0.0, voicing_rate=0.0, frames=0
            ))
            bad += 1
            continue

        s = max(0, int(st * sr))
        e = min(len(y), int(en * sr))
        seg = y[s:e]
        rows.append(compute_f0_stats(seg, sr))

    pros = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), pros], axis=1)
    out.to_csv(out_path, index=False)

    nz = float((out[["f0_mean", "f0_iqr", "voicing_rate", "frames"]].fillna(0) != 0).any(axis=1).mean())
    print(f"[OK] {os.path.basename(csv_path)} <- {os.path.basename(audio_path)} | rows={len(out)} | nonzero={nz:.3f} | bad_time_rows={bad}")


def main():
    parser = argparse.ArgumentParser(description="Batch prosody extraction with exact/stem audio matching.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for CSV files")
    parser.add_argument("--out-suffix", default="_pros", help="Suffix added before .csv")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input CSVs instead of writing new files")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not csv_files:
        print(f"[WARN] No CSVs found in {args.csv_dir} with pattern {args.pattern}")
        return

    for csv_path in csv_files:
        stem = Path(csv_path).stem
        audio_path = resolve_audio(args.audio_dir, stem)
        if audio_path is None:
            print(f"[SKIP] {os.path.basename(csv_path)} -> no matching audio under {args.audio_dir}")
            continue

        if args.overwrite:
            out_path = csv_path
        else:
            out_path = os.path.join(args.csv_dir, stem + args.out_suffix + ".csv")

        process_one(csv_path, audio_path, out_path)


if __name__ == "__main__":
    main()

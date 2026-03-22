#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd


def main(audio_path, in_csv, out_csv):
    try:
        import librosa
    except Exception:
        print("[FATAL] librosa is required. Try: pip install librosa soundfile pandas numpy")
        raise

    if not os.path.exists(audio_path):
        raise SystemExit(f"Missing audio: {audio_path}")
    if not os.path.exists(in_csv):
        raise SystemExit(f"Missing CSV: {in_csv}")

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    df = pd.read_csv(in_csv)

    start_candidates = {"start_sec", "start_s", "start", "onset", "begin"}
    end_candidates = {"end_sec", "end_s", "end", "offset", "finish"}

    start_col = next((c for c in df.columns if c.lower() in start_candidates), None)
    end_col = next((c for c in df.columns if c.lower() in end_candidates), None)

    if start_col is None or end_col is None:
        raise SystemExit(f"Time columns not found. Columns={list(df.columns)}")

    def f0_stats(seg, sr, frame_length=1024, hop_length=160, fmin=60.0, fmax=700.0):
        if len(seg) < hop_length:
            return dict(
                f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
                f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
                f0_slope=0.0, dF0_iqr=0.0, voicing_rate=0.0, frames=0
            )

        f0, _, _ = librosa.pyin(
            seg,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        frames = len(f0)
        if frames == 0:
            return dict(
                f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_iqr=0.0,
                f0_p10=0.0, f0_p90=0.0, f0_range=0.0, f0_cv=0.0,
                f0_slope=0.0, dF0_iqr=0.0, voicing_rate=0.0, frames=0
            )

        voiced_mask = np.isfinite(f0)
        voicing_rate = float(np.mean(voiced_mask)) if frames > 0 else 0.0

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
        t = idx * (hop_length / sr)
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
        rows.append(f0_stats(seg, sr))

    pros = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), pros], axis=1)
    out.to_csv(out_csv, index=False)

    nz = float((out[["f0_mean", "f0_iqr", "voicing_rate", "frames"]].fillna(0) != 0).any(axis=1).mean())
    print(f"[OK] Wrote: {out_csv}")
    print(f"Rows: {len(out)} | bad_time_rows: {bad} | nonzero_fraction: {nz:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/prosody/compute_prosody_one_pair.py /abs/path/to/audio.(wav|mp3) /abs/path/to/input.csv /abs/path/to/output.csv")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

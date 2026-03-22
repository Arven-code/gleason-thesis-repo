from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import parselmouth


FEATURE_COLS = [
    "f0_mean", "f0_median", "f0_std", "f0_iqr", "f0_p10", "f0_p90",
    "f0_range", "f0_cv", "f0_slope", "dF0_iqr", "voicing_rate", "frames"
]


def safe_percentiles(x: np.ndarray, ps: Tuple[int, int]) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    a, b = np.percentile(x, list(ps))
    return float(a), float(b)


def compute_features(seg: parselmouth.Sound, time_step: float, min_pitch: float, max_pitch: float) -> Dict[str, Any]:
    pitch = seg.to_pitch_ac(time_step=time_step, min_pitch=min_pitch, max_pitch=max_pitch)
    f0 = pitch.selected_array["frequency"].astype(float)
    t = np.array(pitch.xs(), dtype=float)

    frames_total = int(f0.size)
    voiced = f0 > 0.0
    voicing_rate = float(np.mean(voiced)) if frames_total > 0 else 0.0

    f0_v = f0[voiced]
    t_v = t[voiced]

    if f0_v.size == 0:
        return {
            "f0_mean": 0.0, "f0_median": 0.0, "f0_std": 0.0, "f0_iqr": 0.0,
            "f0_p10": 0.0, "f0_p90": 0.0, "f0_range": 0.0, "f0_cv": 0.0,
            "f0_slope": 0.0, "dF0_iqr": 0.0, "voicing_rate": voicing_rate, "frames": frames_total
        }

    f0_mean = float(f0_v.mean())
    f0_median = float(np.median(f0_v))
    f0_std = float(f0_v.std(ddof=0))

    p10, p90 = safe_percentiles(f0_v, (10, 90))
    q25, q75 = safe_percentiles(f0_v, (25, 75))
    f0_iqr = float(q75 - q25)
    f0_range = float(p90 - p10)
    f0_cv = float(f0_std / f0_mean) if f0_mean > 0 else 0.0

    if f0_v.size > 1:
        x = t_v - t_v[0]
        f0_slope = float(np.polyfit(x, f0_v, 1)[0])
        dF0 = np.diff(f0_v)
        if dF0.size > 0:
            d25, d75 = safe_percentiles(dF0, (25, 75))
            dF0_iqr = float(d75 - d25)
        else:
            dF0_iqr = 0.0
    else:
        f0_slope = 0.0
        dF0_iqr = 0.0

    return {
        "f0_mean": f0_mean,
        "f0_median": f0_median,
        "f0_std": f0_std,
        "f0_iqr": f0_iqr,
        "f0_p10": p10,
        "f0_p90": p90,
        "f0_range": f0_range,
        "f0_cv": f0_cv,
        "f0_slope": f0_slope,
        "dF0_iqr": dF0_iqr,
        "voicing_rate": voicing_rate,
        "frames": frames_total,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, type=str)
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--out", required=True, type=str)

    p.add_argument("--start-col", default="start_sec", type=str)
    p.add_argument("--end-col", default="end_sec", type=str)

    p.add_argument("--time-step", default=0.01, type=float)
    p.add_argument("--min-pitch", default=75.0, type=float)
    p.add_argument("--max-pitch", default=500.0, type=float)

    p.add_argument("--min-dur", default=0.12, type=float)
    p.add_argument("--log-path", default="", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    df = pd.read_csv(csv_path)

    if args.start_col not in df.columns or args.end_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.start_col}' and '{args.end_col}' columns.")

    snd = parselmouth.Sound(str(audio_path))
    mono = snd.convert_to_mono()

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan

    logs: List[str] = []
    processed = 0
    skipped = 0

    for i, row in df.iterrows():
        start = row[args.start_col]
        end = row[args.end_col]

        if pd.isna(start) or pd.isna(end):
            skipped += 1
            logs.append(f"ROW {i}: SKIP missing time bounds")
            continue

        start_f = float(start)
        end_f = float(end)

        if end_f <= start_f:
            skipped += 1
            logs.append(f"ROW {i}: SKIP invalid bounds start={start_f:.3f} end={end_f:.3f}")
            continue

        dur = end_f - start_f
        if dur < args.min_dur:
            skipped += 1
            logs.append(f"ROW {i}: SKIP too short dur={dur:.3f}s")
            continue

        seg = mono.extract_part(from_time=start_f, to_time=end_f, preserve_times=False)
        if seg.get_total_duration() <= 0:
            skipped += 1
            logs.append(f"ROW {i}: SKIP empty segment")
            continue

        feats = compute_features(
            seg=seg,
            time_step=args.time_step,
            min_pitch=args.min_pitch,
            max_pitch=args.max_pitch,
        )

        for k, v in feats.items():
            df.at[i, k] = v

        processed += 1

    df.to_csv(out_path, index=False)

    if args.log_path:
        Path(args.log_path).write_text("\n".join(logs), encoding="utf-8")

    print(f"Done. Processed={processed}, skipped={skipped}. Output: {out_path}")


if __name__ == "__main__":
    main()

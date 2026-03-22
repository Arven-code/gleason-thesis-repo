# build_glmm_dataset_k5.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--k", default=5, type=int)

    p.add_argument("--speaker-col", default="speaker", type=str)
    p.add_argument("--label-col", default="label", type=str)
    p.add_argument("--session-col", default="session_id", type=str)
    p.add_argument("--child-col", default="child_id", type=str)
    p.add_argument("--turn-col", default="turn_index", type=str)

    p.add_argument("--text-col", default="transcript", type=str)
    return p.parse_args()


def find_prev_caregiver_type(
    speakers: List[str],
    i: int,
    k: int
) -> Optional[str]:
    lo = max(0, i - k)
    for j in range(i - 1, lo - 1, -1):
        sp = speakers[j]
        if sp in ("MOT", "FAT"):
            return sp
    return None


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)

    csv_files = sorted(in_path.rglob("*_training_echo_v2.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_training_echo_v2.csv files found under: {in_path}")

    rows: List[Dict[str, Any]] = []

    for fp in csv_files:
        df = pd.read_csv(fp)

        required = [args.speaker_col, args.label_col, args.session_col, args.child_col, args.turn_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{fp} is missing required columns: {missing}")

        df = df.sort_values([args.session_col, args.turn_col]).reset_index(drop=True)

        for session_id, g in df.groupby(args.session_col, sort=False):
            speakers = g[args.speaker_col].astype(str).tolist()
            labels = g[args.label_col].tolist()
            child_ids = g[args.child_col].tolist()
            turns = g[args.turn_col].tolist()

            texts = g[args.text_col].astype(str).tolist() if args.text_col in g.columns else [""] * len(g)

            for local_i in range(len(g)):
                if speakers[local_i] != "CHI":
                    continue

                caregiver = find_prev_caregiver_type(speakers, local_i, args.k)
                if caregiver is None:
                    continue

                label = float(labels[local_i])
                y_success = 1 if label == 2.0 else 0

                rows.append({
                    "session_id": session_id,
                    "child_id": child_ids[local_i],
                    "turn_index": turns[local_i],
                    "caregiver_type": caregiver,
                    "y_success": y_success,
                    "child_text": texts[local_i],
                    "source_file": str(fp),
                })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Done. Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()

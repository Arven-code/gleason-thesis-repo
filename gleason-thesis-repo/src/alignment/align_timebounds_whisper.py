#!/usr/bin/env python3
# align_timebounds_whisper.py
# Align one transcript CSV to one audio file with Whisper,
# then write start_sec / end_sec for each row into an output CSV.

import os
import re
import argparse
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import whisper


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def best_match_span(query: str, seg_texts: list[str], max_merge: int = 3):
    """
    Return (best_start_index, best_span_len, best_ratio).
    Tries 1, 2, or 3 merged ASR segments.
    """
    best = (-1, 1, 0.0)

    for i in range(len(seg_texts)):
        merged = seg_texts[i]
        ratio = SequenceMatcher(None, query, merged).ratio()
        if ratio > best[2]:
            best = (i, 1, ratio)

        cur = merged
        for span_len in range(2, max_merge + 1):
            if i + span_len - 1 >= len(seg_texts):
                break
            cur = cur + " " + seg_texts[i + span_len - 1]
            ratio = SequenceMatcher(None, query, cur).ratio()
            if ratio > best[2]:
                best = (i, span_len, ratio)

    return best


def align_one(in_csv: str, audio_path: str, out_csv: str, language: str = "en", model_size: str = "small", min_len: float = 0.30):
    in_csv = Path(in_csv)
    audio_path = Path(audio_path)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        raise SystemExit(f"[ERR] Missing CSV: {in_csv}")
    if not audio_path.exists():
        raise SystemExit(f"[ERR] Missing audio: {audio_path}")

    df = pd.read_csv(in_csv)

    if "transcript" not in df.columns:
        if "text" in df.columns:
            df = df.rename(columns={"text": "transcript"})
        else:
            raise SystemExit(f"[ERR] Need a 'transcript' or 'text' column in {in_csv}")

    cache_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)

    model = whisper.load_model(model_size, device="cpu", download_root=cache_dir)
    res = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        fp16=False,
        verbose=False,
        condition_on_previous_text=False,
    )

    segs = []
    for seg in res.get("segments", []):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        segs.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": txt,
                "_norm": normalize_text(txt),
            }
        )

    if not segs:
        raise SystemExit("[ERR] Whisper returned no segments.")

    seg_norms = [s["_norm"] for s in segs]

    starts = []
    ends = []

    for _, row in df.iterrows():
        q = normalize_text(str(row.get("transcript", "")))
        if not q:
            starts.append(np.nan)
            ends.append(np.nan)
            continue

        idx, span_len, ratio = best_match_span(q, seg_norms, max_merge=3)

        if idx < 0:
            starts.append(np.nan)
            ends.append(np.nan)
            continue

        s = segs[idx]["start"]
        e = segs[idx + span_len - 1]["end"]

        if e - s < min_len:
            e = s + min_len

        starts.append(s)
        ends.append(e)

    df["start_sec"] = starts
    df["end_sec"] = ends

    if "label" not in df.columns:
        df["label"] = ""

    df.to_csv(out_csv, index=False)
    print(f"[OK] Aligned -> {out_csv}  rows={len(df)}  with times for {(~pd.isna(df['start_sec'])).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Align one CSV to one audio file using Whisper.")
    parser.add_argument("--audio", required=True, help="Full path to .wav/.mp3")
    parser.add_argument("--in-csv", required=True, help="Full path to input CSV")
    parser.add_argument("--out-csv", required=True, help="Full path to output CSV")
    parser.add_argument("--language", default="en")
    parser.add_argument("--model-size", default="small", choices=["tiny", "base", "small", "medium", "large"])
    args = parser.parse_args()

    align_one(
        in_csv=args.in_csv,
        audio_path=args.audio,
        out_csv=args.out_csv,
        language=args.language,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()

# align_timebounds_whisper_fuzzy.py
# Fill missing start_sec/end_sec in a transcript CSV by aligning rows to Whisper ASR segments
# with fuzzy text overlap (token Jaccard + character n-gram overlap).

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import whisper


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_text(s: str) -> str:
    s = str(s) if s is not None else ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(s))


def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = normalize_text(s)
    s = re.sub(r"\s+", " ", s)
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    tokens: List[str]
    ngrams: List[str]


def build_segments(transcribe_result: Dict, ngram_n: int = 3) -> List[Segment]:
    segs: List[Segment] = []
    for seg in transcribe_result.get("segments", []):
        txt = seg.get("text", "")
        segs.append(
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=txt,
                tokens=tokenize(txt),
                ngrams=char_ngrams(txt, n=ngram_n),
            )
        )
    return segs


def best_match(
    row_text: str,
    segments: List[Segment],
    token_weight: float = 0.6,
    ngram_weight: float = 0.4,
    ngram_n: int = 3,
) -> Tuple[Optional[Segment], float]:
    rtoks = tokenize(row_text)
    rngrams = char_ngrams(row_text, n=ngram_n)

    best_seg: Optional[Segment] = None
    best_score = -1.0

    for seg in segments:
        s_tok = jaccard(rtoks, seg.tokens)
        s_ng = jaccard(rngrams, seg.ngrams)
        score = token_weight * s_tok + ngram_weight * s_ng
        if score > best_score:
            best_score = score
            best_seg = seg

    return best_seg, float(best_score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, type=str)
    p.add_argument("--in-csv", required=True, type=str)
    p.add_argument("--out-csv", required=True, type=str)

    p.add_argument("--language", default="en", type=str)
    p.add_argument("--model-size", default="small", type=str)
    p.add_argument("--temperature", default=0.0, type=float)
    p.add_argument("--no-condition-on-previous-text", action="store_true")

    p.add_argument("--text-col", default="transcript", type=str)
    p.add_argument("--start-col", default="start_sec", type=str)
    p.add_argument("--end-col", default="end_sec", type=str)

    p.add_argument("--ngram-n", default=3, type=int)
    p.add_argument("--token-weight", default=0.6, type=float)
    p.add_argument("--ngram-weight", default=0.4, type=float)
    p.add_argument("--min-score", default=0.15, type=float)

    p.add_argument("--log-path", default="", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    audio_path = Path(args.audio)
    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(in_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column '{args.text_col}' in {in_csv}")
    if args.start_col not in df.columns:
        df[args.start_col] = np.nan
    if args.end_col not in df.columns:
        df[args.end_col] = np.nan

    model = whisper.load_model(args.model_size)

    result = model.transcribe(
        str(audio_path),
        language=args.language,
        temperature=args.temperature,
        condition_on_previous_text=not args.no_condition_on_previous_text,
        fp16=False,
        verbose=False,
    )

    segments = build_segments(result, ngram_n=args.ngram_n)
    if not segments:
        raise RuntimeError("Whisper returned no segments; cannot align time bounds.")

    missing_mask = df[args.start_col].isna() | df[args.end_col].isna()
    missing_idx = df.index[missing_mask].tolist()

    logs: List[str] = []
    filled = 0
    unmatched = 0

    for i in missing_idx:
        row_text = str(df.at[i, args.text_col])
        seg, score = best_match(
            row_text=row_text,
            segments=segments,
            token_weight=args.token_weight,
            ngram_weight=args.ngram_weight,
            ngram_n=args.ngram_n,
        )

        if seg is None or score < args.min_score:
            unmatched += 1
            logs.append(f"ROW {i}: NO MATCH score={score:.3f} text={row_text!r}")
            continue

        df.at[i, args.start_col] = float(seg.start)
        df.at[i, args.end_col] = float(seg.end)
        filled += 1
        logs.append(
            f"ROW {i}: MATCH score={score:.3f} "
            f"start={seg.start:.2f} end={seg.end:.2f} "
            f"row_text={row_text!r} seg_text={seg.text!r}"
        )

    df.to_csv(out_csv, index=False)

    if args.log_path:
        Path(args.log_path).write_text("\n".join(logs), encoding="utf-8")

    print(f"Done. Filled {filled} rows. Unmatched {unmatched} rows. Output: {out_csv}")


if __name__ == "__main__":
    main()

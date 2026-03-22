import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import whisper
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

CACHE_DIR = os.path.expanduser("~/.cache/whisper")

def load_audio(path, sr=16000):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def slice_audio(y, sr, start, end, min_len_s=0.30):
    s = max(0, int(start * sr))
    e = min(len(y), int(end * sr))
    seg = y[s:e]
    need = int(min_len_s * sr)
    if len(seg) < need:
        seg = np.pad(seg, (0, max(0, need - len(seg))))
    return seg

def transcribe_segments(model, audio_path, language="en"):
    res = model.transcribe(
        audio_path,
        fp16=False,
        verbose=False,
        language=language,
        task="transcribe",
        condition_on_previous_text=False,
        temperature=0.0
    )
    segs = []
    for seg in res.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            segs.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text
            })
    return segs

def main(audio_path):
    a = Path(audio_path)
    if not a.exists():
        raise SystemExit(f"Missing audio: {a}")

    out_csv = Path("artifacts") / f"{a.stem}_transcript.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        model = whisper.load_model("small", download_root=CACHE_DIR, device="cpu")
        segs = transcribe_segments(model, str(a), language="en")
        if not segs:
            raise RuntimeError("Empty transcript with small")
    except Exception:
        model = whisper.load_model("base", download_root=CACHE_DIR, device="cpu")
        segs = transcribe_segments(model, str(a), language="en")

    if not segs:
        raise SystemExit("No speech segments found.")

    y, sr = load_audio(str(a), 16000)
    enc = VoiceEncoder()
    embs = []
    for s in segs:
        wav = slice_audio(y, sr, s["start"], s["end"])
        wav_pp = preprocess_wav(wav, source_sr=sr)
        embs.append(enc.embed_utterance(wav_pp))
    X = np.vstack(embs) if embs else np.zeros((0, 256), dtype=np.float32)
    labels = AgglomerativeClustering(n_clusters=2).fit_predict(X) if len(X) >= 2 else np.zeros(len(segs), dtype=int)

    rows = []
    for s, lab in zip(segs, labels):
        rows.append({
            "__source_file": a.name,
            "start_s": round(s["start"], 3),
            "end_s": round(s["end"], 3),
            "speaker": f"SPEAKER_{lab:02d}",
            "text": s["text"]
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/alignment/asr_diarize_simple.py /absolute/path/to/audio.(mp3|wav)")
        sys.exit(2)
    main(sys.argv[1])

import pandas as pd
import re
from difflib import SequenceMatcher
from pathlib import Path

df = pd.read_csv("/mnt/data/william_dataset.csv")

def parse_cha_simple(path):
    utterances = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(FAT|MOT|CHI|RES|INV|MAD|FEM|MAL|PAR|EXP|GRA|GRM|UNK)\s*:\s*(.*)$", line)
            if m:
                spk = m.group(1)
                utt = m.group(2).strip()
                utterances.append((spk, utt))
    return utterances

cha_utts = parse_cha_simple("/mnt/data/william.cha")

def norm(s):
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

cha_texts = [norm(t) for _, t in cha_utts]
cha_speakers = [spk for spk, _ in cha_utts]

speakers_assigned = []
j = 0
misses = 0
threshold = 0.82

for txt in df["transcript"].astype(str).tolist():
    t = norm(txt)
    best_j = None
    best_score = -1.0

    window = 40
    search_range = range(j, min(len(cha_texts), j + window))

    for k in search_range:
        score = SequenceMatcher(a=t, b=cha_texts[k]).ratio()
        if score > best_score:
            best_score = score
            best_j = k
        if best_score == 1.0:
            break

    if best_score >= threshold:
        speakers_assigned.append(cha_speakers[best_j])
        j = best_j + 1
    else:
        search_range = range(j, min(len(cha_texts), j + 200))
        for k in search_range:
            score = SequenceMatcher(a=t, b=cha_texts[k]).ratio()
            if score > best_score:
                best_score = score
                best_j = k
        if best_score >= 0.74:
            speakers_assigned.append(cha_speakers[best_j])
            j = best_j + 1
        else:
            speakers_assigned.append("")
            misses += 1

out = pd.DataFrame({
    "transcript": df["transcript"],
    "label": df.get("label", pd.Series([None] * len(df))),
    "Speaker": speakers_assigned
})

out_path = "/mnt/data/william_with_speaker.csv"
out.to_csv(out_path, index=False)

print(f"Aligned {len(df)} rows. Misses: {misses}")
print(f"Saved to: {out_path}")

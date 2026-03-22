import pandas as pd
import re
from difflib import SequenceMatcher

df = pd.read_csv("/mnt/data/katie_dataset.csv")
tcol = "transcript" if "transcript" in df.columns else ("Transcript" if "Transcript" in df.columns else df.columns[0])

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\x15\d+_\d+\x15", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\(\s*[\d.]+\s*\)", " ", s)
    s = re.sub(r"&[-~]", "", s)
    s = re.sub(r"@[\w]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

cha_utts = []
with open("/mnt/data/katie.cha", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("*"):
            parts = line[1:].split(":", 1)
            if len(parts) != 2:
                continue
            spk = parts[0].strip().upper()
            utt = clean_text(parts[1])
            if utt:
                cha_utts.append((spk, utt))

cha_texts = [u for _, u in cha_utts]
cha_speakers = [s for s, _ in cha_utts]
ds_texts = [clean_text(x) for x in df[tcol].astype(str).tolist()]

speakers_assigned = []
j = 0
misses = 0

for txt in ds_texts:
    best_j = None
    best_score = -1.0

    for k in range(j, min(len(cha_texts), j + 400)):
        score = SequenceMatcher(a=txt, b=cha_texts[k]).ratio()
        if score > best_score:
            best_score = score
            best_j = k
        if best_score >= 0.999:
            break

    if best_score < 0.6:
        for k in range(0, len(cha_texts)):
            score = SequenceMatcher(a=txt, b=cha_texts[k]).ratio()
            if score > best_score:
                best_score = score
                best_j = k

    if best_j is not None and best_score >= 0.6:
        speakers_assigned.append(cha_speakers[best_j])
        j = max(best_j + 1, j)
    else:
        speakers_assigned.append("")
        misses += 1

def is_child_code(code: str) -> bool:
    return bool(re.match(r"^(CHI|CH)\w*$", code))

child_seen = []
mapping = {}
for s in speakers_assigned:
    if s and is_child_code(s) and s not in child_seen:
        child_seen.append(s)

if len(child_seen) <= 1:
    for c in child_seen:
        mapping[c] = "CHI"
else:
    for idx, c in enumerate(child_seen, start=1):
        mapping[c] = f"CHI{idx}"

normalized = [mapping.get(s, s) if s != "" else "" for s in speakers_assigned]

out = pd.DataFrame({
    "transcript": df[tcol],
    "label": df.get("label", pd.Series([None] * len(df))),
    "Speaker": normalized
})

out_path = "/mnt/data/katie_with_speaker.csv"
out.to_csv(out_path, index=False)

print(f"Rows: {len(out)}, misses: {misses}")
print("Detected child codes:", child_seen)
print("Child mapping:", mapping)
print(f"Saved to: {out_path}")

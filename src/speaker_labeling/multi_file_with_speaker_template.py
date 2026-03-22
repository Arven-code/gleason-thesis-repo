import pandas as pd
import re
from difflib import SequenceMatcher

INPUT_CSV = "/mnt/data/bobby_dataset.csv"
INPUT_CHA = "/mnt/data/bobby.cha"
OUTPUT_CSV = "/mnt/data/bobby_with_speaker.csv"

df = pd.read_csv(INPUT_CSV)
cols = df.columns.tolist()
tcol = "transcript" if "transcript" in cols else ("Transcript" if "Transcript" in cols else cols[0])

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\x15\d+_\d+\x15", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"\(\s*[\d.]+\s*\)", " ", s)
    s = re.sub(r"&[-~]", "", s)
    s = re.sub(r"@[\w]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def parse_cha(path: str):
    utts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("*") and ":" in line:
                code, rest = line[1:].split(":", 1)
                spk = code.strip().upper()
                utt = clean_text(rest)
                if utt:
                    utts.append((spk, utt))
            else:
                m = re.match(r"^([A-Z]{2,4}\d?)\s*:\s*(.*)$", line)
                if m:
                    spk = m.group(1).strip().upper()
                    utt = clean_text(m.group(2))
                    if utt:
                        utts.append((spk, utt))
    return utts

cha_utts = parse_cha(INPUT_CHA)
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
        j = max(j, best_j + 1)
    else:
        speakers_assigned.append("")
        misses += 1

def is_child_code(code: str) -> bool:
    return bool(re.match(r"^(CHI|CH)\w*$", code))

child_seen = []
for s in speakers_assigned:
    if s and is_child_code(s) and s not in child_seen:
        child_seen.append(s)

mapping = {}
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

out.to_csv(OUTPUT_CSV, index=False)

print(f"Rows: {len(out)}, misses: {misses}")
print("Child codes detected:", child_seen)
print("Applied mapping:", mapping)
print(f"Saved to: {OUTPUT_CSV}")

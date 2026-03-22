import pandas as pd
import re
from difflib import SequenceMatcher

df = pd.read_csv("/mnt/data/john_dataset.csv")

cha_utts = []
with open("/mnt/data/john.cha", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.startswith("*"):
            try:
                spk, rest = line[1:].split(":", 1)
            except ValueError:
                continue
            spk = spk.strip().upper()
            utt = re.sub(r"\x15\d+_\d+\x15", "", rest).strip()
            cha_utts.append((spk, utt))

def norm(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

cha_texts = [norm(t) for _, t in cha_utts]
cha_speakers = [spk for spk, _ in cha_utts]

speakers_assigned = []
j = 0
misses = 0

for txt in df[df.columns[0]].astype(str).tolist():
    t = norm(txt)
    best_j = None
    best_score = -1.0

    for k in range(j, min(len(cha_texts), j + 60)):
        score = SequenceMatcher(a=t, b=cha_texts[k]).ratio()
        if score > best_score:
            best_score = score
            best_j = k
        if best_score == 1.0:
            break

    if best_score < 0.9:
        for k in range(j, min(len(cha_texts), j + 200)):
            score = SequenceMatcher(a=t, b=cha_texts[k]).ratio()
            if score > best_score:
                best_score = score
                best_j = k

    if best_j is not None and best_score >= 0.75:
        speakers_assigned.append(cha_speakers[best_j])
        j = best_j + 1
    else:
        speakers_assigned.append("")
        misses += 1

child_codes_in_order = []
for s in speakers_assigned:
    if s.startswith("CHI") and s not in child_codes_in_order:
        child_codes_in_order.append(s)

mapping = {}
if len(child_codes_in_order) > 1:
    for idx, code in enumerate(child_codes_in_order, start=1):
        mapping[code] = f"CHI{idx}"

normalized = [mapping.get(s, "CHI" if s == "CHI" else s) for s in speakers_assigned]

out = pd.DataFrame({
    "transcript": df[df.columns[0]],
    "label": df.get("label", pd.Series([None] * len(df))),
    "Speaker": normalized
})

out_path = "/mnt/data/john_with_speaker.csv"
out.to_csv(out_path, index=False)

print(f"Rows: {len(out)}, misses: {misses}, child mapping: {mapping}")
print(f"Saved to: {out_path}")

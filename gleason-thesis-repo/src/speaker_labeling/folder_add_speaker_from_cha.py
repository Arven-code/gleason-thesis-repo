import os
import pandas as pd
import re

cha_folder = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/Father raw .cha scripts"
csv_folder = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/CSV"
output_folder = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Father/Final CSV"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(cha_folder):
    if not file.endswith(".cha"):
        continue

    base = os.path.splitext(file)[0]
    cha_path = os.path.join(cha_folder, file)

    csv_path = None
    for f in os.listdir(csv_folder):
        if f.lower().startswith(base.lower()) and f.lower().endswith(".csv"):
            csv_path = os.path.join(csv_folder, f)
            break

    out_path = os.path.join(output_folder, base + "_with_speaker.csv")

    if csv_path is None or not os.path.exists(csv_path):
        print(f"❌ CSV not found for {base}, skipping.")
        continue

    with open(cha_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    speaker_list = []
    for line in lines:
        m = re.match(r"^\*(FAT|CHI|STF|RES):\t(.+)", line)
        if m:
            speaker = m.group(1).replace("STF", "RES")
            speaker_list.append(speaker)

    df = pd.read_csv(csv_path)

    if len(speaker_list) < len(df):
        print(f"⚠️ Not enough lines in {base}.cha to match {base}.csv — assigning only available speakers.")
        speaker_list += [None] * (len(df) - len(speaker_list))

    df["Speaker"] = speaker_list[:len(df)]
    df.to_csv(out_path, index=False)
    print(f"✅ {base}: Speaker labels added. Saved to {out_path}")

import os
import re
import pandas as pd

cha_folder = "/Users/arvendobay/.../cha_folder"
csv_folder = "/Users/arvendobay/.../clean_for_labeling"
output_folder = "/Users/arvendobay/.../labeled_output"
os.makedirs(output_folder, exist_ok=True)

speaker_re = re.compile(r"^(MOT|FAT|CHI|SAR):\s*(.+)", re.IGNORECASE)

for cha_fname in os.listdir(cha_folder):
    if not cha_fname.lower().endswith(".cha"):
        continue

    base = os.path.splitext(cha_fname)[0]
    cha_path = os.path.join(cha_folder, cha_fname)

    csv_fname = next(
        (f for f in os.listdir(csv_folder)
         if f.lower().startswith(base.lower()) and f.lower().endswith(".csv")),
        None
    )
    if not csv_fname:
        print(f"⚠️ No CSV for {base}, skipping.")
        continue

    df = pd.read_csv(os.path.join(csv_folder, csv_fname))
    print(f"{base}: read {len(df)} rows from {csv_fname}")

    speakers = []
    with open(cha_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = speaker_re.match(line)
            if m:
                code = m.group(1).upper()
                if code == "SAR":
                    code = "RES"
                speakers.append(code)

    print(f"{base}: extracted {len(speakers)} speaker tags from .cha")

    df["label"] = [
        speakers[i] if i < len(speakers) else "UNKNOWN"
        for i in range(len(df))
    ]

    out_path = os.path.join(output_folder, f"{base}_with_speaker.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ {base}: saved labeled CSV → {out_path}")

print("🎉 All done.")

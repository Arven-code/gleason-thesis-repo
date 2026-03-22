# Add an empty 'label' column to each cleaned CSV file
# so the child utterances can be labeled by hand before BERT training.

import pandas as pd
from pathlib import Path

input_dir = Path("/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE/MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Dinner/cleaned")
output_dir = input_dir.parent / "for_labeling"
output_dir.mkdir(exist_ok=True)

for csv_file in input_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)
    if "label" not in df.columns:
        df["label"] = ""
    output_file = output_dir / csv_file.name.replace("_cleaned.csv", "_for_labeling.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ {output_file.name} written")

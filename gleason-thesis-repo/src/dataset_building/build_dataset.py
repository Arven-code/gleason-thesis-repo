import os
import pandas as pd

def process_folder(folder):
    rows = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file)) as f:
                for line in f:
                    if ":" in line:
                        speaker, text = line.split(":", 1)
                        rows.append({
                            "Speaker": speaker.strip(),
                            "transcript": text.strip()
                        })
    return pd.DataFrame(rows)

df = process_folder("INPUT_FOLDER")
df.to_csv("output.csv", index=False)

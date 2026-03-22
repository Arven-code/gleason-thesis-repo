import re

input_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/andy.cha"
output_file = "/Users/arvendobay/Desktop/Documents/Documents 2025/UNINE /MA/Comms/MA thesis/CHILDES/Eng-NA/Gleason/Mother/andy.txt"

pattern = re.compile(r'^\*?([A-Z]+):\s*(.*)')

dialogue_lines = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        match = pattern.match(line)
        if match:
            speaker, utterance = match.groups()
            dialogue_lines.append(utterance)

with open(output_file, "w", encoding="utf-8") as f:
    for utterance in dialogue_lines:
        f.write(utterance + "\n")

print(f"Converted {input_file} to {output_file}.")

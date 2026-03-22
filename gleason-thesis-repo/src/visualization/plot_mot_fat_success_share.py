import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "artifacts/mlm_mot_fat/success_preceder_overall.csv"
)

mot = 30
fat = 37
total = mot + fat

mot_pct = mot / total * 100
fat_pct = fat / total * 100

plt.figure(figsize=(5, 5))
plt.bar(["MOT", "FAT"], [mot_pct, fat_pct])

plt.ylabel("Percentage of successes (%)")
plt.title("Child success by preceding caregiver")

for i, v in enumerate([mot_pct, fat_pct]):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")

plt.savefig(
    "artifacts/mlm_mot_fat/mot_fat_success_composition.png"
)
plt.close()

pd.DataFrame({
    "group": ["MOT", "FAT"],
    "percentage": [mot_pct, fat_pct]
}).to_csv(
    "artifacts/mlm_mot_fat/mot_fat_success_composition.csv",
    index=False
)

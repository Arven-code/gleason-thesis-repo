import pandas as pd
from math import comb

path = "/mnt/data/gender_ids_group_table.csv"

df = pd.read_csv(path)

required = {"group", "success", "N", "rate"}
assert required.issubset(set(df.columns)), f"Columns present: {df.columns.tolist()}"

mot = df[df["group"] == "MOT"].iloc[0]
fat = df[df["group"] == "FAT"].iloc[0]

a = int(mot["success"])
b = int(mot["N"] - mot["success"])
c = int(fat["success"])
d = int(fat["N"] - fat["success"])

row1, row2 = a + b, c + d
col1, col2 = a + c, b + d
n = row1 + row2

def pmf(k):
    if k < max(0, col1 - row2) or k > min(col1, row1):
        return 0.0
    return comb(col1, k) * comb(col2, row1 - k) / comb(n, row1)

p_obs = pmf(a)
p_two = sum(
    pmf(k)
    for k in range(max(0, col1 - row2), min(col1, row1) + 1)
    if pmf(k) <= p_obs + 1e-15
)

aa, bb, cc, dd = a, b, c, d
if 0 in (aa, bb, cc, dd):
    aa += 0.5
    bb += 0.5
    cc += 0.5
    dd += 0.5
odds_ratio = (aa * dd) / (bb * cc)

mot_rate = a / (a + b) if (a + b) > 0 else float("nan")
fat_rate = c / (c + d) if (c + d) > 0 else float("nan")
delta_pp = 100 * (mot_rate - fat_rate)

paragraph = (
    f"Across all files, child success following MOT turns was {mot_rate*100:.1f}% ({a}/{a+b}) "
    f"versus {fat_rate*100:.1f}% ({c}/{c+d}) after FAT turns; difference = {delta_pp:.1f} pp "
    f"(Fisher’s exact: OR={odds_ratio:.2f}, p={p_two:.3g})."
)

print("a,b,c,d =", a, b, c, d)
print("mot_rate,fat_rate,delta_pp =", mot_rate, fat_rate, delta_pp)
print("paragraph:", paragraph)

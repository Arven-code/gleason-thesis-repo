import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import comb, sqrt, log, exp

tbl_path = "/mnt/data/gender_ids_group_table.csv"
df = pd.read_csv(tbl_path)

mot = df[df["group"] == "MOT"].iloc[0]
fat = df[df["group"] == "FAT"].iloc[0]

a = int(mot["success"])
b = int(mot["N"] - mot["success"])
c = int(fat["success"])
d = int(fat["N"] - fat["success"])

p_mot = a / (a + b)
p_fat = c / (c + d)

def wilson_ci(x, n, z=1.96):
    phat = x / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half

ci_mot = wilson_ci(a, a + b)
ci_fat = wilson_ci(c, c + d)

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

OR = (a * d) / (b * c)
se_logOR = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
or_low = exp(log(OR) - 1.96 * se_logOR)
or_high = exp(log(OR) + 1.96 * se_logOR)

color_mot = "#2c7bb6"
color_fat = "#6a51a3"
grid_color = "#e5e7eb"

def save_fig(path_png, path_pdf, dark=False):
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=240)

    labels = ["MOT (29/145)", "FAT (38/241)"]
    y = np.arange(2)

    rates = np.array([p_mot, p_fat]) * 100
    lo = np.array([ci_mot[0], ci_fat[0]]) * 100
    hi = np.array([ci_mot[1], ci_fat[1]]) * 100
    cols = [color_mot, color_fat]

    if dark:
        fig.patch.set_facecolor("#0b1220")
        ax.set_facecolor("#0b1220")
        grid_c = "#1f2a44"
        text_c = "#e5e7eb"
        edge_c = "white"
        ci_alpha = 0.45
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        grid_c = grid_color
        text_c = "#111827"
        edge_c = "white"
        ci_alpha = 0.35

    for yi, l, h, col in zip(y, lo, hi, cols):
        ax.plot([l, h], [yi, yi], "-", lw=5.5, color=col, alpha=ci_alpha, solid_capstyle="round")

    ax.scatter(rates, y, s=200, color=cols, edgecolors=edge_c, linewidths=1.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5, color=text_c)
    ax.set_xlabel("Success rate (%)", fontsize=12, color=text_c)
    ax.set_xlim(0, max(hi.max() * 1.15, 30))
    ax.grid(axis="x", color=grid_c, linewidth=1, alpha=1.0)
    ax.set_axisbelow(True)

    for yi, r, l, h in zip(y, rates, lo, hi):
        ax.text(h + 1.0, yi, f"[{l:.1f}, {h:.1f}]", va="center", ha="left", fontsize=10.5, color=text_c)

    delta = (p_mot - p_fat) * 100
    title = "Child success after the caregiver’s turn (MOT vs FAT)"
    subtitle = f"Δ = {delta:.1f} pp; OR = {OR:.2f} (95% CI {or_low:.2f}–{or_high:.2f}); Fisher’s p = {p_two:.3f}"
    ax.set_title(title, loc="left", fontsize=14.5, pad=10, color=text_c)
    ax.text(0.0, -0.22, subtitle, transform=ax.transAxes, fontsize=11, ha="left", va="top", color=text_c)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path_png, facecolor=fig.get_facecolor(), edgecolor="none")
    fig.savefig(path_pdf, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

light_png = "/mnt/data/MOT_FAT_success_bentz_light_v2.png"
light_pdf = "/mnt/data/MOT_FAT_success_bentz_light_v2.pdf"
dark_png = "/mnt/data/MOT_FAT_success_bentz_dark_v2.png"
dark_pdf = "/mnt/data/MOT_FAT_success_bentz_dark_v2.pdf"

save_fig(light_png, light_pdf, dark=False)
save_fig(dark_png, dark_pdf, dark=True)

print("Saved:")
print(light_png)
print(light_pdf)
print(dark_png)
print(dark_pdf)
print(f"Δ = {(p_mot - p_fat)*100:.1f} pp; OR={OR:.2f}; p={p_two:.3f}")

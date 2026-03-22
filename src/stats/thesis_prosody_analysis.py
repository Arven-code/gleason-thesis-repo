import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ART = BASE / "artifacts"
INFILE = ART / "train_rows_for_glmm_prosody.csv"
OR_CSV = ART / "glmm_continuous_prosody_effects_py.csv"
FIG_FOREST = ART / "fig_or_forest_py.png"
FIG_ME = ART / "fig_me_z_f0_iqr_py.png"
FIG_INT = ART / "fig_interaction_f0iqr_sex_py.png"

ART.mkdir(parents=True, exist_ok=True)

if not INFILE.exists():
    raise SystemExit(f"Missing input: {INFILE}")

df = pd.read_csv(INFILE)

grp_col = "child_id" if "child_id" in df.columns else ("__source_file" if "__source_file" in df.columns else None)
if grp_col is None or "label_acc" not in df.columns:
    raise SystemExit("Need child_id or __source_file, and label_acc in the CSV.")

sex_map = {
    1: "mother", 0: "father",
    True: "mother", False: "father",
    "m": "mother", "f": "father",
    "mother": "mother", "father": "father"
}

if "caregiver_sex" in df.columns:
    care = df["caregiver_sex"].map(sex_map)
    care = care.fillna(df["caregiver_sex"].astype("string"))
else:
    care = pd.Series(pd.NA, index=df.index, dtype="string")

for key in ("__source_dir", "__source_file"):
    if key in df.columns:
        s = df[key].astype("string").str.lower()
        care = care.mask(care.isna() & s.str.contains("mother", na=False), "mother")
        care = care.mask(care.isna() & s.str.contains("father", na=False), "father")

care = care.astype("string").str.strip().str.lower()
df["sex_mother"] = np.where(care == "mother", 1.0, np.where(care == "father", 0.0, np.nan))
has_sex = pd.notna(df["sex_mother"]).any()

df = df[(df["voicing_rate"].fillna(0) >= 0.3) & (df["frames"].fillna(0) >= 30)].copy()

def wins(x, p=0.01):
    x = x.astype(float)
    lo, hi = np.nanquantile(x, p), np.nanquantile(x, 1 - p)
    return np.clip(x, lo, hi)

pros_cols = ["f0_mean","f0_iqr","f0_range","f0_slope","voicing_rate","frames","dF0_iqr","f0_cv"]
for c in [c for c in pros_cols if c in df.columns]:
    df[c] = wins(df[c], 0.01)

def zscore(x):
    x = x.astype(float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    sd = 1.0 if (not np.isfinite(sd) or sd == 0) else sd
    return (x - mu) / sd

for c in [c for c in pros_cols if c in df.columns]:
    df["z_" + c] = zscore(df[c])

X_cols = [c for c in ["z_f0_mean","z_f0_iqr","z_f0_range","z_f0_slope","z_voicing_rate","z_frames","z_dF0_iqr","z_f0_cv"] if c in df.columns]
if not X_cols:
    raise SystemExit("No predictor columns found after QC/z-scoring.")

X = pd.DataFrame(index=df.index)
X["const"] = 1.0
for c in X_cols:
    X[c] = df[c]

if has_sex:
    df["sex_mother"] = df["sex_mother"].fillna(0.0)
    X["sex_mother"] = df["sex_mother"]
    if "z_f0_iqr" in X.columns:
        X["z_f0_iqr_x_sex"] = X["z_f0_iqr"] * X["sex_mother"]

y = df["label_acc"].astype(int)

model = sm.GLM(y, X, family=sm.families.Binomial())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[grp_col]})

params = res.params
conf = res.conf_int()
or_df = pd.DataFrame({
    "term": params.index,
    "estimate": params.values,
    "std_error": np.sqrt(np.diag(res.cov_params())),
    "OR": np.exp(params.values),
    "OR_low": np.exp(conf[0].values),
    "OR_high": np.exp(conf[1].values),
})
or_df = or_df[or_df["term"] != "const"].copy()
or_df.to_csv(OR_CSV, index=False)

plt.figure(figsize=(7, 4))
order = or_df.sort_values("OR")["term"]
vals = or_df.set_index("term").loc[order]
ypos = np.arange(len(order))
plt.axvline(1.0, linestyle="--")
plt.errorbar(vals["OR"], ypos,
             xerr=[vals["OR"] - vals["OR_low"], vals["OR_high"] - vals["OR"]],
             fmt='o', capsize=3)
plt.yticks(ypos, order)
plt.xscale("log")
plt.xlabel("Odds ratio (log scale)")
plt.title("Prosody predictors of child accuracy (95% CI)")
plt.tight_layout()
plt.savefig(FIG_FOREST, dpi=300)
plt.close()

if "z_f0_iqr" in X.columns:
    grid = np.linspace(-2.5, 2.5, 60)
    base = X.mean().to_frame().T
    Xgrid = pd.concat([base] * len(grid), ignore_index=True)
    Xgrid["z_f0_iqr"] = grid
    if "z_f0_iqr_x_sex" in Xgrid.columns:
        Xgrid["sex_mother"] = 0.0
        Xgrid["z_f0_iqr_x_sex"] = Xgrid["z_f0_iqr"] * Xgrid["sex_mother"]
    pred = res.predict(Xgrid)
    plt.figure(figsize=(6, 4))
    plt.plot(grid, pred)
    plt.xlabel("Pitch spread (z F0_IQR)")
    plt.ylabel("Predicted Pr(accuracy)")
    plt.title("Marginal effect of caregiver pitch spread")
    plt.tight_layout()
    plt.savefig(FIG_ME, dpi=300)
    plt.close()

if "z_f0_iqr" in X.columns and "sex_mother" in X.columns:
    grid = np.linspace(-2.5, 2.5, 60)
    base = X.mean().to_frame().T
    curves = []
    for sex_val, label in [(0.0, "Father"), (1.0, "Mother")]:
        Xg = pd.concat([base] * len(grid), ignore_index=True)
        Xg["z_f0_iqr"] = grid
        Xg["sex_mother"] = sex_val
        if "z_f0_iqr_x_sex" in Xg.columns:
            Xg["z_f0_iqr_x_sex"] = Xg["z_f0_iqr"] * Xg["sex_mother"]
        curves.append((label, grid, res.predict(Xg)))
    plt.figure(figsize=(6, 4))
    for label, g, pr in curves:
        plt.plot(g, pr, label=label)
    plt.xlabel("Pitch spread (z F0_IQR)")
    plt.ylabel("Predicted Pr(accuracy)")
    plt.title("Pitch spread × caregiver sex")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_INT, dpi=300)
    plt.close()

print("Saved:")
print(" -", OR_CSV)
print(" -", FIG_FOREST)
if FIG_ME.exists():
    print(" -", FIG_ME)
if FIG_INT.exists():
    print(" -", FIG_INT)

#!/usr/bin/env python3
import os, glob, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ART = "artifacts"

def main():
    candidates = [
        os.path.join(ART, "train_rows_for_glmm_prosody.csv"),
        os.path.join(ART, "train_rows_for_glmm_textonly.csv"),
        os.path.join(ART, "train_rows_for_glmm_shufflePROS.csv"),
    ]
    files = [f for f in candidates if os.path.exists(f)]
    if not files:
        files = glob.glob(os.path.join(ART, "train_rows_for_glmm_*.csv"))
    if not files:
        raise SystemExit(
            "No GLMM input found under artifacts/. "
            "Run: python src/models/train_prosody_mlm.py (without --no-prosody)."
        )
    infile = files[0]
    print(f"[INFO] Using input: {infile}")

    df = pd.read_csv(infile)

    grp_col = "child_id" if "child_id" in df.columns else ("__source_file" if "__source_file" in df.columns else None)
    if grp_col is None or "label_acc" not in df.columns:
        raise SystemExit("Need child_id or __source_file and label_acc columns in the CSV.")

    pros_cols = ["f0_mean","f0_std","f0_min","f0_max","f0_median","f0_iqr","f0_slope","voicing_rate","frames"]
    missing = [c for c in pros_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing prosody columns: {missing}")

    all_zero = (df[pros_cols] == 0).sum(axis=1) == len(pros_cols)
    if all_zero.any():
        df = df.loc[~all_zero].copy()

    def wins(x, p=0.005):
        lo = np.nanquantile(x, p); hi = np.nanquantile(x, 1 - p)
        return np.clip(x, lo, hi)

    for c in pros_cols:
        df[c] = wins(df[c].astype(float))

    df["f0_cv"] = np.where(df["f0_mean"] == 0, np.nan, df["f0_std"] / df["f0_mean"])

    zsrc = ["f0_mean","f0_iqr","f0_slope","voicing_rate","frames","f0_cv"]
    for c in zsrc:
        if c in df.columns:
            mu, sd = df[c].mean(), df[c].std(ddof=0)
            sd = 1.0 if (not np.isfinite(sd) or sd == 0) else sd
            df["z_" + c] = (df[c] - mu) / sd

    X_cols = [c for c in ["z_f0_mean","z_f0_iqr","z_f0_slope","z_voicing_rate","z_frames","z_f0_cv"] if c in df.columns]
    if not X_cols:
        raise SystemExit("No standardized predictors available.")
    X = sm.add_constant(df[X_cols])
    y = df["label_acc"].astype(int)

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df[grp_col]})

    params = res.params
    conf = res.conf_int()
    out = pd.DataFrame({
        "term": params.index,
        "estimate": params.values,
        "std_error": np.sqrt(np.diag(res.cov_params())),
        "OR": np.exp(params.values),
        "OR_low": np.exp(conf[0].values),
        "OR_high": np.exp(conf[1].values),
    })
    out = out[out["term"] != "const"]

    os.makedirs(ART, exist_ok=True)
    csv_out = os.path.join(ART, "glmm_continuous_prosody_effects_py.csv")
    out.to_csv(csv_out, index=False)

    order = out.sort_values("OR")["term"]
    vals = out.set_index("term").loc[order]
    plt.figure()
    plt.axvline(1.0, linestyle="--")
    ypos = np.arange(len(order))
    plt.errorbar(vals["OR"], ypos, xerr=[vals["OR"] - vals["OR_low"], vals["OR_high"] - vals["OR"]], fmt='o')
    plt.yticks(ypos, order)
    plt.xscale("log")
    plt.xlabel("Odds ratio (log scale)")
    plt.title("Prosody predictors of child accuracy (GLM w/ cluster-robust CI)")
    png_out = os.path.join(ART, "glmm_continuous_prosody_effects_py.png")
    plt.tight_layout(); plt.savefig(png_out, dpi=200)

    print("[OK] Saved:")
    print(" -", csv_out)
    print(" -", png_out)

if __name__ == "__main__":
    main()

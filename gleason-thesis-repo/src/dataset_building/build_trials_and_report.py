#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_trials_and_report.py

Build trials.csv by pairing each CHI row (label in {1,2}) with the immediately
preceding caregiver row (MOT/FAT). Then compute MOT vs FAT success and IDS usage
if an ids flag column exists, and write a paragraph plus summary tables.
"""
import argparse, os, sys, glob, json, math
from typing import Optional, List
import pandas as pd
import numpy as np

FATHER_TOKENS = {"FAT","FTH","FATHER","FATHER.","DAD","DADA","PAPA","PA","F","FATH","FTR","FAR","PÈRE","PERE","PÉRE"}
MOTHER_TOKENS = {"MOT","MOTHER","MOTHER.","MOM","MUM","MAMA","MAMAN","MA","M"}

def detect_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def norm_group(val: str) -> Optional[str]:
    s = str(val).upper().strip()
    if s in MOTHER_TOKENS or s.startswith("MOT"):
        return "MOT"
    if s in FATHER_TOKENS or s.startswith(("FAT", "FTH", "FATH")):
        return "FAT"
    return None

def parse_label_to_y(v):
    if pd.isna(v):
        return np.nan
    try:
        x = float(str(v).strip())
        if x == 2.0:
            return 1
        if x == 1.0:
            return 0
    except:
        pass
    s = str(v).strip().lower()
    if s in {"2", "success", "succ"}:
        return 1
    if s in {"1", "fail"}:
        return 0
    return np.nan

def fisher_exact_2x2(a, b, c, d):
    from math import comb
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
    return (aa * dd) / (bb * cc), p_two

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not files:
        print(f"ERROR: no CSV files in {args.in_dir}", file=sys.stderr)
        sys.exit(2)

    trials = []
    diag = {"unknown_speaker_tokens": {}, "files_processed": 0}

    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Skip unreadable: {p} ({e})", file=sys.stderr)
            continue

        spk_col = detect_col(df, ["Speaker", "speaker", "caregiver", "caregiver_group"])
        if spk_col is None:
            continue
        lbl_col = detect_col(df, ["label", "child_label", "y"])
        if lbl_col is None:
            continue
        start_col = detect_col(df, ["start_sec", "start", "start_time", "start_ms"])

        if start_col:
            df = df.sort_values(start_col, kind="mergesort").reset_index(drop=True)

        spk_norm = df[spk_col].astype(str).str.upper().str.strip()

        for tok, n in spk_norm.value_counts().items():
            if tok not in MOTHER_TOKENS and tok not in FATHER_TOKENS and tok not in {"CHI"}:
                diag["unknown_speaker_tokens"][tok] = diag["unknown_speaker_tokens"].get(tok, 0) + int(n)

        for i, row in df.iterrows():
            if str(spk_norm.iat[i]) != "CHI":
                continue

            y = parse_label_to_y(row[lbl_col])
            if np.isnan(y):
                continue

            j = i - 1
            care = None
            ids_flag = np.nan

            while j >= 0:
                g = norm_group(spk_norm.iat[j])
                if g in {"MOT", "FAT"}:
                    care = g
                    ids_col = detect_col(df, ["ids_like", "ids", "ids_flag", "idslike"])
                    if ids_col is not None:
                        try:
                            v = df[ids_col].iat[j]
                            if pd.isna(v):
                                ids_flag = 0
                            else:
                                s = str(v).strip().lower()
                                ids_flag = 1 if s in {"1", "true", "t", "yes", "y"} else (
                                    0 if s in {"0", "false", "f", "no", "n"}
                                    else int(round(float(s))) if s.replace(".", "", 1).isdigit() else 0
                                )
                        except:
                            ids_flag = 0
                    break
                j -= 1

            if care is None:
                continue

            trials.append({
                "file": os.path.basename(p),
                "caregiver_group": care,
                "y": int(y),
                "ids_like": (int(ids_flag) if not np.isnan(ids_flag) else "")
            })

        diag["files_processed"] += 1

    if not trials:
        print("ERROR: no trials constructed (check that CHI rows have label=1/2).", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    trials_df = pd.DataFrame(trials)
    trials_df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    g = trials_df.groupby("caregiver_group")["y"].agg(["sum", "count"]).rename(columns={"sum": "succ", "count": "N"})
    for k in ["MOT", "FAT"]:
        if k not in g.index:
            g.loc[k] = [0, 0]
    g = g.loc[["MOT", "FAT"]]
    g["rate"] = g["succ"] / g["N"]

    a = int(g.loc["MOT", "succ"])
    b = int(g.loc["MOT", "N"] - a)
    c = int(g.loc["FAT", "succ"])
    d = int(g.loc["FAT", "N"] - c)
    OR, p = fisher_exact_2x2(a, b, c, d)
    delta_pp = (g.loc["MOT", "rate"] - g.loc["FAT", "rate"]) * 100.0

    per_file = trials_df.groupby(["file", "caregiver_group"])["y"].mean().reset_index()
    n_files = int(trials_df["file"].nunique())
    mot_med = float(per_file.query("caregiver_group=='MOT'")["y"].median()) if (per_file["caregiver_group"] == "MOT").any() else float("nan")
    mot_iqr = float(per_file.query("caregiver_group=='MOT'")["y"].quantile(0.75) - per_file.query("caregiver_group=='MOT'")["y"].quantile(0.25)) if (per_file["caregiver_group"] == "MOT").any() else float("nan")
    fat_med = float(per_file.query("caregiver_group=='FAT'")["y"].median()) if (per_file["caregiver_group"] == "FAT").any() else float("nan")
    fat_iqr = float(per_file.query("caregiver_group=='FAT'")["y"].quantile(0.75) - per_file.query("caregiver_group=='FAT'")["y"].quantile(0.25)) if (per_file["caregiver_group"] == "FAT").any() else float("nan")

    has_ids = "ids_like" in trials_df.columns and trials_df["ids_like"].notna().any() and (trials_df["ids_like"] != "").any()
    ids_rate_mot = ids_rate_fat = succ_ids = succ_nonids = most_ids = None
    if has_ids:
        trials_df["_ids"] = pd.to_numeric(trials_df["ids_like"], errors="coerce").fillna(0).astype(int)
        ids_by_group = trials_df.groupby("caregiver_group")["_ids"].mean()
        ids_rate_mot = float(ids_by_group.get("MOT", np.nan))
        ids_rate_fat = float(ids_by_group.get("FAT", np.nan))
        succ_ids = float(trials_df.loc[trials_df["_ids"] == 1, "y"].mean())
        succ_nonids = float(trials_df.loc[trials_df["_ids"] == 0, "y"].mean())
        if not math.isnan(ids_rate_mot) and not math.isnan(ids_rate_fat):
            most_ids = "MOT" if ids_rate_mot >= ids_rate_fat else "FAT"

    def pct(x):
        return f"{100.0 * float(x):.1f}%"

    mot_rate = float(g.loc["MOT", "rate"]) if g.loc["MOT", "N"] > 0 else float("nan")
    fat_rate = float(g.loc["FAT", "rate"]) if g.loc["FAT", "N"] > 0 else float("nan")
    mot_s, mot_N = int(g.loc["MOT", "succ"]), int(g.loc["MOT", "N"])
    fat_s, fat_N = int(g.loc["FAT", "succ"]), int(g.loc["FAT", "N"])

    parts = [f"Across {n_files} files, child success following MOT turns was {pct(mot_rate)} ({mot_s}/{mot_N}) versus {pct(fat_rate)} ({fat_s}/{fat_N}) after FAT turns; difference = {delta_pp:.1f} pp (Fisher’s exact: OR={OR:.2f}, p={p:.3g})."]
    if not (math.isnan(mot_med) or math.isnan(fat_med)):
        parts.append(f"At the file level, the median success rate was {pct(mot_med)} (IQR {pct(mot_iqr)}) for MOT and {pct(fat_med)} (IQR {pct(fat_iqr)}) for FAT.")
    if has_ids:
        parts.append(f"IDS-like (‘motherese’) prosody occurred more often in {most_ids} ({pct(ids_rate_mot)} MOT vs {pct(ids_rate_fat)} FAT). Within IDS-flagged spans, success was {pct(succ_ids)} versus {pct(succ_nonids)} in non-IDS spans.")
    paragraph = " ".join(parts)

    pd.DataFrame({"group": ["MOT", "FAT"], "success": [mot_s, fat_s], "N": [mot_N, fat_N], "rate": [mot_rate, fat_rate]}).to_csv(
        os.path.join(out_dir, "gender_ids_group_table.csv"), index=False)

    with open(os.path.join(out_dir, "gender_ids_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "MOT": {"succ": mot_s, "N": mot_N, "rate": mot_rate},
            "FAT": {"succ": fat_s, "N": fat_N, "rate": fat_rate},
            "delta_pp": delta_pp,
            "fisher": {"odds_ratio": OR, "p_two_sided": p},
            "n_files": n_files,
            "per_file": {"MOT_median_rate": mot_med, "MOT_IQR": mot_iqr, "FAT_median_rate": fat_med, "FAT_IQR": fat_iqr},
            "ids": {"available": has_ids, "MOT_rate": ids_rate_mot, "FAT_rate": ids_rate_fat, "succ_in_IDS": succ_ids, "succ_in_nonIDS": succ_nonids, "most_IDS_group": most_ids},
            "diag": diag
        }, f, indent=2)
    with open(os.path.join(out_dir, "gender_ids_report.txt"), "w", encoding="utf-8") as f:
        f.write(paragraph + "\n")

    print("\n=== Ready-to-paste paragraph ===\n")
    print(paragraph)
    print("\nSaved to:", out_dir)
    print("Trials:", os.path.join(out_dir, "trials.csv"))

if __name__ == "__main__":
    main()

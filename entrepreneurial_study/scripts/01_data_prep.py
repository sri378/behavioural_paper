"""
01_data_prep.py
===============
Data Preparation Pipeline for Entrepreneurial Barriers & Education Study
Author  : ML Pipeline (Research Paper)
Purpose : Load raw data, rename columns, run reliability checks,
          detect outliers, and export analysis_ready.csv

Run:    python scripts/01_data_prep.py
Output: data/processed/analysis_ready.csv
        outputs/tables/reliability_report.csv
        outputs/tables/outlier_report.csv
"""

import os, sys
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.stats import chi2, pearsonr
from factor_analyzer import ConfirmatoryFactorAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH   = os.path.join(BASE_DIR, "data", "raw", "Entrepreneurial_Education_data.xlsx")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
TABLE_DIR  = os.path.join(BASE_DIR, "outputs", "tables")
FIG_DIR    = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: LOADING RAW DATA")
print("=" * 60)

df_raw = pd.read_excel(RAW_PATH)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns: {df_raw.columns.tolist()}")

# ─────────────────────────────────────────────
# RENAME COLUMNS  (canonical short names)
# ─────────────────────────────────────────────
rename_map = {
    "Moderator" : "MOD",
    "COG 1": "C1", "COG 2": "C2", "COG 3": "C3", "COG 4": "C4", "COG 5": "C5",
    "REG 1": "R1", "REG 2": "R2", "REG 3": "R3", "REG 4": "R4", "REG 5": "R5",
    "NORM 1": "N1", "NORM 2": "N2", "NORM 3": "N3", "NORM 4": "N4",
    "NORM 5": "N5", "NORM 6": "N6",
    "INTENT 1": "I1", "INTENT 2": "I2", "INTENT 3": "I3",
    "INTENT 4": "I4", "INTENT 5": "I5", "INTENT 6": "I6",
    "MEAN COG"   : "MEAN_COG",
    "MEAN REG"   : "MEAN_REG",
    "MEAN NORM"  : "MEAN_NORM",
    "MEAN INTENT": "MEAN_INT",
}
df = df_raw.rename(columns=rename_map).copy()

# Moderator: 1 = Yes (done module), 2 = No → recode to 1/0
# MOD_bin = 1 means student HAS done the entrepreneurship module
df["MOD_bin"] = (df["MOD"] == 1).astype(int)
df["MOD_label"] = df["MOD_bin"].map({1: "EduYes", 0: "EduNo"})

print(f"\nModerator distribution:")
print(df["MOD_label"].value_counts())

# ─────────────────────────────────────────────
# DEFINE ITEM GROUPS
# ─────────────────────────────────────────────
COG_ITEMS    = ["C1", "C2", "C3", "C4", "C5"]
REG_ITEMS    = ["R1", "R2", "R3", "R4", "R5"]
NORM_ITEMS   = ["N1", "N2", "N3", "N4", "N5", "N6"]
INTENT_ITEMS = ["I1", "I2", "I3", "I4", "I5", "I6"]
ALL_ITEMS    = COG_ITEMS + REG_ITEMS + NORM_ITEMS + INTENT_ITEMS

print("\n" + "=" * 60)
print("STEP 2: MISSING DATA & RANGE CHECK")
print("=" * 60)

missing_df = (
    df[ALL_ITEMS].isna().sum()
    .to_frame("missing_n")
    .assign(missing_pct=lambda x: (100 * x["missing_n"] / len(df)).round(2))
)
print(missing_df)

range_rows = []
for c in ALL_ITEMS:
    out_of_range = int(((df[c] < 1) | (df[c] > 6)).sum())
    range_rows.append({"item": c, "min": df[c].min(), "max": df[c].max(),
                        "out_of_range": out_of_range})
range_df = pd.DataFrame(range_rows)
print("\nRange check (Likert 1–6):")
print(range_df[range_df["out_of_range"] > 0] if range_df["out_of_range"].any() else "All items within 1–6. ✓")
range_df.to_csv(os.path.join(TABLE_DIR, "range_check.csv"), index=False)

# ─────────────────────────────────────────────
# STEP 3: RELIABILITY (CRONBACH'S ALPHA)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: CRONBACH'S ALPHA")
print("=" * 60)

rel_rows = []
for name, items in [("COG", COG_ITEMS), ("REG", REG_ITEMS),
                     ("NORM", NORM_ITEMS), ("INTENT", INTENT_ITEMS)]:
    alpha, ci = pg.cronbach_alpha(data=df[items])
    print(f"  {name:8s}: α = {alpha:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    rel_rows.append({"construct": name, "n_items": len(items),
                     "alpha": round(alpha, 3),
                     "ci_lower": round(ci[0], 3), "ci_upper": round(ci[1], 3)})

rel_df = pd.DataFrame(rel_rows)
rel_df.to_csv(os.path.join(TABLE_DIR, "reliability_report.csv"), index=False)
print(f"\n  Saved → {TABLE_DIR}/reliability_report.csv")

# ─────────────────────────────────────────────
# STEP 4: INTER-ITEM CORRELATIONS (heatmap)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: INTER-ITEM CORRELATION HEATMAP")
print("=" * 60)

corr_matrix = df[ALL_ITEMS].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.3, ax=ax,
            annot_kws={"size": 7})
ax.set_title("Inter-item Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_inter_item_corr.png"), dpi=150)
plt.close()
print(f"  Saved → {FIG_DIR}/01_inter_item_corr.png")

# ─────────────────────────────────────────────
# STEP 5: MAHALANOBIS OUTLIER DETECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: MAHALANOBIS DISTANCE OUTLIER DETECTION")
print("=" * 60)

X = df[ALL_ITEMS].copy().values
mu  = X.mean(axis=0)
cov = np.cov(X, rowvar=False)
try:
    cov_inv = inv(cov)
except np.linalg.LinAlgError:
    cov_inv = np.linalg.pinv(cov)

md  = np.array([mahalanobis(row, mu, cov_inv) for row in X])
md2 = md ** 2
df["Mahalanobis"] = md
df["Mahalanobis_sq"] = md2

threshold_sq = chi2.ppf(0.999, df=len(ALL_ITEMS))
outliers = df[df["Mahalanobis_sq"] > threshold_sq].copy()
outlier_idx = outliers.index.tolist()

print(f"  χ² threshold (p<.001, df={len(ALL_ITEMS)}): {threshold_sq:.2f}")
print(f"  Outliers detected: {len(outlier_idx)}")
print(f"  Outlier indices  : {outlier_idx}")

outliers[["Mahalanobis", "Mahalanobis_sq"]].sort_values(
    "Mahalanobis_sq", ascending=False
).to_csv(os.path.join(TABLE_DIR, "outlier_report.csv"))
print(f"  Saved → {TABLE_DIR}/outlier_report.csv")

# Keep full dataset (flag outliers) — researcher decides whether to remove
df["is_outlier"] = df["Mahalanobis_sq"] > threshold_sq
df_clean = df[~df["is_outlier"]].copy().reset_index(drop=True)

print(f"\n  Full   : {df.shape[0]} rows")
print(f"  Clean  : {df_clean.shape[0]} rows  (outliers flagged but both saved)")

# ─────────────────────────────────────────────
# STEP 6: BUILD COMPOSITE SCORES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: COMPOSITE SCORES (row means)")
print("=" * 60)

for dataset, label in [(df, "full"), (df_clean, "clean")]:
    for name, items in [("COG", COG_ITEMS), ("REG", REG_ITEMS),
                         ("NORM", NORM_ITEMS), ("INT", INTENT_ITEMS)]:
        dataset[f"{name}_mean"] = dataset[items].mean(axis=1)

    # z-score standardisation
    for col in ["COG_mean", "REG_mean", "NORM_mean", "INT_mean"]:
        dataset[col + "_z"] = (dataset[col] - dataset[col].mean()) / dataset[col].std()

print("  Composite means and z-scores added.")

# ─────────────────────────────────────────────
# STEP 7: CONSTRUCT CORRELATION TABLE
# ─────────────────────────────────────────────
constructs = ["COG_mean", "REG_mean", "NORM_mean", "INT_mean"]
construct_labels = ["COG", "REG", "NORM", "INT"]
corr_constructs = df_clean[constructs].corr()
corr_constructs.index = construct_labels
corr_constructs.columns = construct_labels
corr_constructs.to_csv(os.path.join(TABLE_DIR, "construct_correlations.csv"))
print(f"\n  Construct correlation table saved.")
print(corr_constructs.round(3))

# ─────────────────────────────────────────────
# STEP 8: EXPORT CSVs
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: EXPORTING PROCESSED DATA")
print("=" * 60)

FULL_OUT  = os.path.join(PROC_DIR, "analysis_ready_full.csv")
CLEAN_OUT = os.path.join(PROC_DIR, "analysis_ready_clean.csv")

df.to_csv(FULL_OUT, index=False)
df_clean.to_csv(CLEAN_OUT, index=False)

print(f"  Full  dataset → {FULL_OUT}")
print(f"  Clean dataset → {CLEAN_OUT}")
print(f"\n✓ Data prep complete. Shape (clean): {df_clean.shape}")
print("=" * 60)

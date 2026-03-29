import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold, cross_validate, train_test_split, cross_val_score
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# PATHS
BASE      = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "processed", "analysis_ready_clean.csv")
TABLE_DIR = os.path.join(BASE, "outputs", "tables")
FIG_DIR   = os.path.join(BASE, "outputs", "figures")

# LOAD DATA
print("=" * 60)
print("LOADING PROCESSED DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"Groups: {df['MOD_label'].value_counts().to_dict()}")


# DEFINE FEATURES

COG_ITEMS    = ["C1","C2","C3","C4","C5"]
REG_ITEMS    = ["R1","R2","R3","R4","R5"]
NORM_ITEMS   = ["N1","N2","N3","N4","N5","N6"]
INTENT_ITEMS = ["I1","I2","I3","I4","I5","I6"]

# Composite means (already in df from step 01)
X_cols = ["COG_mean", "REG_mean", "NORM_mean"]
y_col  = "INT_mean"
MOD    = "MOD_bin"   # 1 = has done module, 0 = hasn't

X = df[X_cols].copy()
y = df[y_col].copy()
mod = df[MOD].copy()

print(f"\nFeatures : {X_cols}")
print(f"Target   : {y_col}")
print(f"Moderator: {MOD}  (1=EduYes, 0=EduNo)")


# STEP 1: STANDARDISED OLS REGRESSION
print("\n" + "=" * 60)
print("STEP 1: STANDARDISED OLS REGRESSION")
print("=" * 60)

scaler = StandardScaler()
X_z = pd.DataFrame(scaler.fit_transform(X), columns=[c+"_z" for c in X_cols])
y_z = (y - y.mean()) / y.std()

X_sm = sm.add_constant(X_z)
ols_model = sm.OLS(y_z, X_sm).fit()
print(ols_model.summary())

std_betas = pd.DataFrame({
    "Predictor"     : ["COG (Cognitive)", "REG (Regulative)", "NORM (Normative)"],
    "Std_Beta"      : ols_model.params[1:].values,
    "SE"            : ols_model.bse[1:].values,
    "t"             : ols_model.tvalues[1:].values,
    "p_value"       : ols_model.pvalues[1:].values,
    "CI_lower"      : ols_model.conf_int()[0][1:].values,
    "CI_upper"      : ols_model.conf_int()[1][1:].values,
}).sort_values("Std_Beta", ascending=False)

print("\n── Predictor Ranking (Standardised β) ──")
print(std_betas.to_string(index=False))
std_betas.to_csv(os.path.join(TABLE_DIR, "ols_standardised_betas.csv"), index=False)

# OLS beta plot
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#d62728" if b == std_betas["Std_Beta"].max() else "#1f77b4"
          for b in std_betas["Std_Beta"]]
ax.barh(std_betas["Predictor"], std_betas["Std_Beta"], color=colors,
        xerr=1.96 * std_betas["SE"], capsize=4)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_xlabel("Standardised β")
ax.set_title("OLS: Standardised Effects on Entrepreneurial Intention", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_ols_betas.png"), dpi=150)
plt.close()


# STEP 2: CROSS-VALIDATED MODEL COMPARISON=
print("\n" + "=" * 60)
print("STEP 2: CROSS-VALIDATED MODEL COMPARISON (5-fold CV)")
print("=" * 60)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = ("r2", "neg_mean_squared_error", "neg_mean_absolute_error")

models = {
    "Linear Regression" : Pipeline([("imp", SimpleImputer()), ("mdl", LinearRegression())]),
    "Ridge"             : Pipeline([("imp", SimpleImputer()), ("mdl", Ridge(alpha=1.0))]),
    "Random Forest"     : Pipeline([("imp", SimpleImputer()),
                          ("mdl", RandomForestRegressor(n_estimators=500,
                                                         min_samples_leaf=3,
                                                         random_state=42))]),
    "Gradient Boosting" : Pipeline([("imp", SimpleImputer()),
                          ("mdl", GradientBoostingRegressor(n_estimators=300,
                                                             learning_rate=0.05,
                                                             max_depth=3,
                                                             random_state=42))]),
}

cv_results = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring,
                            return_train_score=False)
    cv_results.append({
        "Model": name,
        "R2_mean" : round(scores["test_r2"].mean(), 4),
        "R2_std"  : round(scores["test_r2"].std(), 4),
        "RMSE"    : round((-scores["test_neg_mean_squared_error"].mean()) ** 0.5, 4),
        "MAE"     : round(-scores["test_neg_mean_absolute_error"].mean(), 4),
    })
    print(f"  {name:22s}  R²={scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")

cv_df = pd.DataFrame(cv_results).sort_values("R2_mean", ascending=False)
cv_df.to_csv(os.path.join(TABLE_DIR, "cv_model_comparison.csv"), index=False)
print(f"\n  Best model: {cv_df.iloc[0]['Model']}")


# STEP 3: RANDOM FOREST — full model + SHAP

print("\n" + "=" * 60)
print("STEP 3: RANDOM FOREST + PERMUTATION IMPORTANCE + SHAP")
print("=" * 60)

X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
    X, y, mod, test_size=0.25, random_state=42, stratify=mod
)

rf_model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=3,
                                  max_features="sqrt", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
print(f"  RMSE : {mean_squared_error(y_test, y_pred)**0.5:.4f}")
print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")

# Permutation importance
perm = permutation_importance(rf_model, X_test, y_test,
                              n_repeats=100, random_state=42, scoring="r2")
perm_df = pd.DataFrame({
    "Feature"         : X_cols,
    "Importance_mean" : perm.importances_mean,
    "Importance_std"  : perm.importances_std,
}).sort_values("Importance_mean", ascending=False)
perm_df["Short"] = ["COG (Cognitive)","REG (Regulative)","NORM (Normative)"]
print("\n── Permutation Importance ──")
print(perm_df.to_string(index=False))
perm_df.to_csv(os.path.join(TABLE_DIR, "permutation_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4))
colors_p = ["#d62728" if i == 0 else "#1f77b4" for i in range(3)]
ax.barh(perm_df["Short"], perm_df["Importance_mean"],
        xerr=perm_df["Importance_std"], color=colors_p, capsize=4)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_xlabel("Mean ΔR² (permutation)")
ax.set_title("Random Forest: Permutation Feature Importance", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_permutation_importance.png"), dpi=150)
plt.close()

# SHAP
print("\n  Computing SHAP values...")
explainer    = shap.TreeExplainer(rf_model)
shap_values  = explainer.shap_values(X_test)
shap_df      = pd.DataFrame(shap_values, columns=X_cols)
shap_mean    = shap_df.abs().mean().sort_values(ascending=False)
print("\n── SHAP Mean |value| ──")
print(shap_mean)
shap_df.to_csv(os.path.join(TABLE_DIR, "shap_values_full.csv"), index=False)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, feature_names=X_cols,
                  show=False, plot_type="dot")
plt.title("SHAP Summary Plot — Full Sample", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# SHAP bar plot (mean |SHAP|)
fig, ax = plt.subplots(figsize=(7, 4))
labels = {"COG_mean":"COG (Cognitive)", "REG_mean":"REG (Regulative)", "NORM_mean":"NORM (Normative)"}
colors_s = ["#d62728" if i == 0 else "#1f77b4"
            for i in range(len(shap_mean))]
ax.barh([labels.get(f, f) for f in shap_mean.index], shap_mean.values,
        color=colors_s)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP: Global Feature Importance", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_shap_bar.png"), dpi=150)
plt.close()


# STEP 4: MODERATION — STRATIFIED ANALYSIS (by education group)

print("\n" + "=" * 60)
print("STEP 4a: MODERATION — STRATIFIED RF+SHAP")
print("=" * 60)

strat_results = {}
shap_by_group = {}

for grp_val, grp_name in [(1, "EduYes"), (0, "EduNo")]:
    mask = (df["MOD_bin"] == grp_val)
    Xg   = df.loc[mask, X_cols].reset_index(drop=True)
    yg   = df.loc[mask, y_col].reset_index(drop=True)

    if len(Xg) < 30:
        print(f"  ⚠ {grp_name}: too few samples ({len(Xg)}), skipping.")
        continue

    # OLS betas
    Xg_z = pd.DataFrame(StandardScaler().fit_transform(Xg), columns=[c+"_z" for c in X_cols])
    yg_z = (yg - yg.mean()) / yg.std()
    ols_g = sm.OLS(yg_z, sm.add_constant(Xg_z)).fit()

    # RF + SHAP
    rf_g = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42)
    rf_g.fit(Xg, yg)
    explainer_g = shap.TreeExplainer(rf_g)
    shap_g = explainer_g.shap_values(Xg)
    shap_mean_g = np.abs(shap_g).mean(axis=0)

    cv_r2 = cross_val_score(rf_g, Xg, yg, cv=KFold(5, shuffle=True, random_state=42),
                             scoring="r2").mean()

    strat_results[grp_name] = {
        "n"           : len(Xg),
        "OLS_COG_beta": round(ols_g.params.iloc[1], 4),
        "OLS_REG_beta": round(ols_g.params.iloc[2], 4),
        "OLS_NOR_beta": round(ols_g.params.iloc[3], 4),
        "OLS_R2"      : round(ols_g.rsquared, 4),
        "RF_CV_R2"    : round(cv_r2, 4),
        "SHAP_COG"    : round(shap_mean_g[0], 4),
        "SHAP_REG"    : round(shap_mean_g[1], 4),
        "SHAP_NORM"   : round(shap_mean_g[2], 4),
    }
    shap_by_group[grp_name] = {"values": shap_g, "X": Xg}

    print(f"\n  Group: {grp_name}  (n={len(Xg)})")
    print(f"    OLS R²   : {ols_g.rsquared:.4f}")
    print(f"    RF CV-R² : {cv_r2:.4f}")
    print(f"    Std Betas: COG={ols_g.params.iloc[1]:.3f}  REG={ols_g.params.iloc[2]:.3f}  NORM={ols_g.params.iloc[3]:.3f}")
    print(f"    SHAP |μ| : COG={shap_mean_g[0]:.3f}  REG={shap_mean_g[1]:.3f}  NORM={shap_mean_g[2]:.3f}")

strat_df = pd.DataFrame(strat_results).T
strat_df.to_csv(os.path.join(TABLE_DIR, "moderation_stratified.csv"))
print(f"\n  Saved → outputs/tables/moderation_stratified.csv")

# ── Comparison plot ──
if len(strat_results) == 2:
    groups     = ["EduYes", "EduNo"]
    predictors = ["COG", "REG", "NORM"]
    shap_cols  = ["SHAP_COG", "SHAP_REG", "SHAP_NORM"]

    y_pos = np.arange(len(predictors))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: OLS std betas
    beta_cols = ["OLS_COG_beta","OLS_REG_beta","OLS_NOR_beta"]
    for i, (grp, col, mk) in enumerate(
            zip(groups, [beta_cols]*2, ["o","s"])):
        vals = [strat_df.loc[grp, c] for c in beta_cols]
        axes[0].barh(y_pos + (i*w - w/2), vals, w, label=grp,
                     color=["#2c7bb6","#d7191c"][i], alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(predictors)
    axes[0].axvline(0, color="black", lw=0.8, linestyle="--")
    axes[0].set_xlabel("Standardised β")
    axes[0].set_title("OLS: Std Betas by Education Group")
    axes[0].legend()

    # Right: SHAP
    for i, grp in enumerate(groups):
        vals = [strat_df.loc[grp, c] for c in shap_cols]
        axes[1].barh(y_pos + (i*w - w/2), vals, w, label=grp,
                     color=["#2c7bb6","#d7191c"][i], alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(predictors)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("SHAP Importance by Education Group")
    axes[1].legend()

    plt.suptitle("Moderation by Entrepreneurial Education",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_moderation_comparison.png"), dpi=150)
    plt.close()
    print("  Moderation comparison plot saved.")

# STEP 4b: MODERATION — INTERACTION TERM REGRESSION

print("\n" + "=" * 60)
print("STEP 4b: MODERATION — INTERACTION TERM REGRESSION")
print("=" * 60)

# Standardise all predictors
X_int = X.copy()
X_int["MOD"] = mod.values
y_z_all = (y - y.mean()) / y.std()

# Standardise continuous predictors
for col in X_cols:
    X_int[col] = (X_int[col] - X_int[col].mean()) / X_int[col].std()
# Moderator mean-centered
X_int["MOD_c"] = X_int["MOD"] - X_int["MOD"].mean()

# Create interaction terms
for col in X_cols:
    short = col.replace("_mean","")
    X_int[f"{short}_x_MOD"] = X_int[col] * X_int["MOD_c"]

int_cols = X_cols + ["MOD_c"] + [c.replace("_mean","")+"_x_MOD" for c in X_cols]
X_int_sm = sm.add_constant(X_int[int_cols])
ols_int   = sm.OLS(y_z_all, X_int_sm).fit()

print(ols_int.summary())

# Extract interaction significance
int_params = ols_int.params.filter(regex="_x_MOD")
int_pvals  = ols_int.pvalues.filter(regex="_x_MOD")
int_ci     = ols_int.conf_int().filter(regex="_x_MOD", axis=0)

int_result = pd.DataFrame({
    "Interaction"      : int_params.index,
    "Beta"             : int_params.values.round(4),
    "p_value"          : int_pvals.values.round(4),
    "CI_lower"         : int_ci[0].values.round(4),
    "CI_upper"         : int_ci[1].values.round(4),
    "Significant"      : (int_pvals.values < 0.05),
})
print("\n── Interaction Terms (Moderation Effects) ──")
print(int_result.to_string(index=False))
int_result.to_csv(os.path.join(TABLE_DIR, "moderation_interaction_terms.csv"), index=False)


# STEP 4c: BOOTSTRAP MODERATION TEST

print("\n" + "=" * 60)
print("STEP 4c: BOOTSTRAP MODERATION TEST (B=2000)")
print("=" * 60)

def bootstrap_moderation(X, y, mod, n_boot=2000, predictor_col="COG_mean", random_state=42):
    """Bootstrap difference in std beta for one predictor across two groups."""
    rng   = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_boot):
        idx   = rng.choice(len(X), size=len(X), replace=True)
        Xb    = X.iloc[idx].reset_index(drop=True)
        yb    = y.iloc[idx].reset_index(drop=True)
        modb  = mod.iloc[idx].reset_index(drop=True)

        betas = []
        for grp in [1, 0]:
            mask_g = (modb == grp)
            if mask_g.sum() < 10:
                betas.append(np.nan); continue
            Xg = Xb.loc[mask_g, [predictor_col]]
            yg = yb[mask_g]
            Xg_z = (Xg - Xg.mean()) / Xg.std()
            yg_z = (yg - yg.mean()) / yg.std()
            mdl  = sm.OLS(yg_z, sm.add_constant(Xg_z)).fit()
            betas.append(mdl.params.iloc[1])
        diffs.append(betas[0] - betas[1])  # EduYes − EduNo

    diffs = np.array([d for d in diffs if not np.isnan(d)])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": diffs.mean(), "ci_lower": ci_lo, "ci_upper": ci_hi,
            "p_approx": 2 * min((diffs > 0).mean(), (diffs < 0).mean())}

boot_rows = []
for pred_col, pred_label in zip(X_cols, ["COG (Cognitive)","REG (Regulative)","NORM (Normative)"]):
    res = bootstrap_moderation(X, y, mod, n_boot=2000, predictor_col=pred_col)
    res["Predictor"] = pred_label
    boot_rows.append(res)
    print(f"  {pred_label:20s}: Δβ = {res['mean_diff']:+.4f}  "
          f"95% CI [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]  "
          f"p≈{res['p_approx']:.4f}")

boot_df = pd.DataFrame(boot_rows)[["Predictor","mean_diff","ci_lower","ci_upper","p_approx"]]
boot_df.columns = ["Predictor","Mean_Delta_Beta","CI_lower","CI_upper","p_approx"]
boot_df.to_csv(os.path.join(TABLE_DIR, "bootstrap_moderation.csv"), index=False)
print(f"\n  Bootstrap results saved.")

# Bootstrap forest plot
fig, ax = plt.subplots(figsize=(8, 4))
xerr_lo = (boot_df["Mean_Delta_Beta"] - boot_df["CI_lower"]).values
xerr_hi = (boot_df["CI_upper"]        - boot_df["Mean_Delta_Beta"]).values
ax.errorbar(
    boot_df["Mean_Delta_Beta"].values,
    boot_df["Predictor"].values,
    xerr=np.array([xerr_lo, xerr_hi]),
    fmt="o", markersize=8, capsize=5, color="#2c7bb6"
)
ax.axvline(0, color="red", linestyle="--", lw=1.2)
ax.set_xlabel("β (EduYes) − β (EduNo)  [Bootstrap 95% CI]")
ax.set_title("Moderation Test: Education Effect on Each Predictor's β",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_bootstrap_moderation.png"), dpi=150)
plt.close()


# STEP 5: ROBUSTNESS CHECK — no-outlier sample

print("\n" + "=" * 60)
print("STEP 5: ROBUSTNESS CHECK (full vs no-outlier)")
print("=" * 60)

df_full = pd.read_csv(os.path.join(BASE, "data", "processed", "analysis_ready_full.csv"))
X_f = df_full[X_cols]; y_f = df_full[y_col]
X_fz = pd.DataFrame(StandardScaler().fit_transform(X_f), columns=[c+"_z" for c in X_cols])
y_fz = (y_f - y_f.mean()) / y_f.std()
ols_f = sm.OLS(y_fz, sm.add_constant(X_fz)).fit()

robust_df = pd.DataFrame({
    "Predictor"      : ["COG","REG","NORM"],
    "Beta_clean"     : ols_model.params[1:].round(4).values,
    "Beta_full"      : ols_f.params[1:].round(4).values,
    "p_clean"        : ols_model.pvalues[1:].round(4).values,
    "p_full"         : ols_f.pvalues[1:].round(4).values,
})
print(robust_df.to_string(index=False))
robust_df.to_csv(os.path.join(TABLE_DIR, "robustness_full_vs_clean.csv"), index=False)


# STEP 6: CONSOLIDATED SUMMARY TABLE

print("\n" + "=" * 60)
print("STEP 6: CONSOLIDATED SUMMARY TABLE")
print("=" * 60)

summary = pd.DataFrame({
    "Predictor"          : ["COG (Cognitive)", "REG (Regulative)", "NORM (Normative)"],
    "OLS_Std_Beta"       : ols_model.params[1:].round(4).values,
    "OLS_p"              : ols_model.pvalues[1:].round(4).values,
    "SHAP_mean_abs"      : [shap_mean.get(c, 0) for c in X_cols],
    "Perm_Importance"    : perm_df.set_index("Feature")["Importance_mean"][X_cols].round(4).values,
    "Rank_OLS"           : pd.Series(ols_model.params[1:].values).rank(ascending=False).values.astype(int),
    "Rank_SHAP"          : pd.Series([shap_mean.get(c, 0) for c in X_cols]).rank(ascending=False).values.astype(int),
})
print(summary.to_string(index=False))
summary.to_csv(os.path.join(TABLE_DIR, "final_summary_table.csv"), index=False)

print("\n" + "=" * 60)
print("✓ ML ANALYSIS COMPLETE")
print("=" * 60)
print("\nOutputs saved to outputs/tables/ and outputs/figures/")
print("\nKey tables:")
print("  ols_standardised_betas.csv      — predictor ranking")
print("  permutation_importance.csv      — RF feature importance")
print("  shap_values_full.csv            — SHAP per observation")
print("  moderation_stratified.csv       — group-wise analysis")
print("  moderation_interaction_terms.csv— interaction regressions")
print("  bootstrap_moderation.csv        — bootstrap CI for Δβ")
print("  robustness_full_vs_clean.csv    — outlier sensitivity")
print("  final_summary_table.csv         — consolidated results")

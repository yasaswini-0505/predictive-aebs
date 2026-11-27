import os
import sys
from pathlib import Path
import inspect

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, brier_score_loss
)

# 0. Setup Paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# 1. Load Model
model_path = SCRIPT_DIR / "aeb_random_forest_model.pkl"
model = joblib.load(model_path)
print("================================================================")
print("             AEB MODEL & PIPELINE INSPECTION                    ")
print("================================================================")
print(f"Model path: {model_path.name}")
print(f"Pipeline steps: {[name for name, _ in model.steps]}")

preproc = model.named_steps["preprocessor"]
rf = model.named_steps["classifier"]
print(f"Random Forest configuration: n_estimators={rf.n_estimators}, random_state={rf.random_state}")
for trans in preproc.transformers_:
    print(f"  Transformer: {trans[0]}, columns: {trans[2]}")
    if hasattr(trans[1], "categories_"):
        print(f"    Categories: {trans[1].categories_}")

# 2. Check Feature Consistency
print("\n================================================================")
print("             FEATURE CONSISTENCY CHECK                          ")
print("================================================================")
df = pd.read_csv(SCRIPT_DIR / "aeb_dataset.csv")
X = df.drop("collision_risk", axis=1)
y = df["collision_risk"]
print(f"Training dataset columns (N={len(X.columns)}): {list(X.columns)}")

from aeb_agent import AEBAgent
agent = AEBAgent()
print(f"AEBAgent inference columns (N={len(agent.FEATURE_COLUMNS)}): {agent.FEATURE_COLUMNS}")
features_match = (list(X.columns) == agent.FEATURE_COLUMNS)
print(f"Exact column name and order match? {features_match}")

# 3. Data Leakage, Index Disjointness, and Duplication Checks
print("\n================================================================")
print("       DATA INTEGRITY, SPLIT DISJOINTNESS & LEAKAGE CHECKS      ")
print("================================================================")
print(f"Total rows in aeb_dataset.csv: {len(df)}")
dups_in_train = df.duplicated().sum()
print(f"Duplicate rows in aeb_dataset.csv: {dups_in_train}")

new_df = pd.read_csv(SCRIPT_DIR / "aeb_new_dataset.csv")
print(f"Total rows in aeb_new_dataset.csv: {len(new_df)}")
dups_in_new = new_df.duplicated().sum()
print(f"Duplicate rows in aeb_new_dataset.csv: {dups_in_new}")

# Exact feature-value duplicate check between datasets
overlap_datasets = pd.merge(df, new_df, on=list(X.columns), how="inner")
print(f"Exact feature-identical rows between original and new datasets: {len(overlap_datasets)}")

# Perform the exact 80/20 train/test split from save_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Rigorous split disjointness verification by dataset row indices
train_indices = set(X_train.index)
test_indices = set(X_test.index)
index_overlap = train_indices.intersection(test_indices)
print(f"\nTrain/Test Index Verification:")
print(f"  Training split size : {len(train_indices)} rows ({len(train_indices)/len(df)*100:.1f}%)")
print(f"  Test split size     : {len(test_indices)} rows ({len(test_indices)/len(df)*100:.1f}%)")
print(f"  Shared row indices  : {len(index_overlap)} (Strictly disjoint: {len(index_overlap) == 0})")
print(f"  Partition coverage  : {len(train_indices) + len(test_indices)} / {len(df)} rows")

# Content-level check: do any distinct rows share identical features across splits?
content_overlap = pd.merge(
    df.iloc[list(train_indices)], df.iloc[list(test_indices)],
    on=list(X.columns), how="inner"
)
print(f"  Content-level identical feature rows across train/test splits: {len(content_overlap)}")

# 4. Evaluate on Held-Out Test Set (2,000 samples)
print("\n================================================================")
print("     EVALUATION ON HELD-OUT TEST SPLIT (2,000 samples)          ")
print("================================================================")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)
roc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print(f"Confusion Matrix [TN, FP; FN, TP]:")
print(f"  TN: {tn:<5}  FP: {fp:<5}")
print(f"  FN: {fn:<5}  TP: {tp:<5}")
print(f"Accuracy               : {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision (PPV)        : {prec:.4f} ({prec*100:.2f}%)")
print(f"Recall (Sensitivity)   : {rec:.4f} ({rec*100:.2f}%)")
print(f"Specificity (TNR)      : {tn / (tn + fp):.4f} ({tn / (tn + fp)*100:.2f}%)")
print(f"F1-Score               : {f1:.4f}")
print(f"False Negative Rate    : {fnr:.4f} ({fnr*100:.2f}%) [{fn}/{fn+tp} missed hazards]")
print(f"False Positive Rate    : {fpr:.4f} ({fpr*100:.2f}%) [{fp}/{fp+tn} false alarms]")
print(f"ROC AUC                : {roc:.4f}")
print(f"Brier Score            : {brier:.4f}")

# Evaluate on aeb_new_dataset.csv (1,000 samples)
print("\n================================================================")
print("     EVALUATION ON INDEPENDENT NEW DATASET (1,000 samples)      ")
print("================================================================")
X_new = new_df.drop("collision_risk", axis=1)
y_new = new_df["collision_risk"]
y_new_pred = model.predict(X_new)
y_new_prob = model.predict_proba(X_new)[:, 1]

cm_new = confusion_matrix(y_new, y_new_pred)
tn_n, fp_n, fn_n, tp_n = cm_new.ravel()
acc_n = accuracy_score(y_new, y_new_pred)
prec_n = precision_score(y_new, y_new_pred)
rec_n = recall_score(y_new, y_new_pred)
f1_n = f1_score(y_new, y_new_pred)
fnr_n = fn_n / (fn_n + tp_n)
fpr_n = fp_n / (fp_n + tn_n)
roc_n = roc_auc_score(y_new, y_new_prob)

print(f"Confusion Matrix [TN, FP; FN, TP]:")
print(f"  TN: {tn_n:<5}  FP: {fp_n:<5}")
print(f"  FN: {fn_n:<5}  TP: {tp_n:<5}")
print(f"Accuracy               : {acc_n:.4f} ({acc_n*100:.2f}%)")
print(f"Precision (PPV)        : {prec_n:.4f} ({prec_n*100:.2f}%)")
print(f"Recall (Sensitivity)   : {rec_n:.4f} ({rec_n*100:.2f}%)")
print(f"F1-Score               : {f1_n:.4f}")
print(f"False Negative Rate    : {fnr_n:.4f} ({fnr_n*100:.2f}%) [{fn_n}/{fn_n+tp_n} missed hazards]")
print(f"False Positive Rate    : {fpr_n:.4f} ({fpr_n*100:.2f}%) [{fp_n}/{fp_n+tn_n} false alarms]")
print(f"ROC AUC                : {roc_n:.4f}")

# 5. Extract Specific False Negative and False Positive Cases
test_analysis_df = X_test.copy()
test_analysis_df["actual"] = y_test
test_analysis_df["predicted"] = y_pred
test_analysis_df["prob_collision"] = y_prob

fn_cases = test_analysis_df[(test_analysis_df["actual"] == 1) & (test_analysis_df["predicted"] == 0)]
fp_cases = test_analysis_df[(test_analysis_df["actual"] == 0) & (test_analysis_df["predicted"] == 1)]

print("\n================================================================")
print(f"   FALSE NEGATIVE EXAMPLES (Actual Hazard=1, Predicted=0) [N={len(fn_cases)}]")
print("================================================================")
print(fn_cases.head(8).to_string())

print("\n================================================================")
print(f"   FALSE POSITIVE EXAMPLES (Actual Safe=0, Predicted=1) [N={len(fp_cases)}]")
print("================================================================")
print(fp_cases.head(8).to_string())

# 6. Error Analysis Across Feature Slices
print("\n================================================================")
print("             ERROR ANALYSIS ACROSS FEATURE SLICES               ")
print("================================================================")

def slice_report(df_all, feature_col, bins=None, labels=None):
    temp = df_all.copy()
    if bins is not None:
        temp["slice"] = pd.cut(temp[feature_col], bins=bins, labels=labels)
    else:
        temp["slice"] = temp[feature_col]
    
    grouped = temp.groupby("slice", observed=False).agg(
        total=("actual", "count"),
        actual_pos=("actual", lambda s: (s == 1).sum()),
        fn=("actual", lambda s: ((s == 1) & (temp.loc[s.index, "predicted"] == 0)).sum()),
        fp=("actual", lambda s: ((s == 0) & (temp.loc[s.index, "predicted"] == 1)).sum()),
    )
    grouped["fnr_pct"] = (grouped["fn"] / grouped["actual_pos"] * 100).fillna(0).round(2)
    grouped["fpr_pct"] = (grouped["fp"] / (grouped["total"] - grouped["actual_pos"]) * 100).fillna(0).round(2)
    return grouped

print("\n--- By Road Environment ---")
print(slice_report(test_analysis_df, "road_environment"))

print("\n--- By Weather Condition ---")
print(slice_report(test_analysis_df, "weather_condition"))

print("\n--- By Friction Range ---")
print(slice_report(test_analysis_df, "friction", bins=[0.0, 0.3, 0.7, 1.0], labels=["Icy (0-0.3)", "Wet (0.3-0.7)", "Dry (0.7-1.0)"]))

print("\n--- By Vehicle Speed ---")
print(slice_report(test_analysis_df, "vehicle_speed", bins=[0, 40, 70, 110], labels=["Low (<40)", "Medium (40-70)", "High (>70)"]))

print("\n--- By Distance to Obstacle ---")
print(slice_report(test_analysis_df, "distance", bins=[0, 15, 30, 60], labels=["Short (<15m)", "Medium (15-30m)", "Long (>30m)"]))

print("\n--- By Relative Speed ---")
print(slice_report(test_analysis_df, "relative_speed", bins=[0, 25, 50, 110], labels=["Low (<25)", "Medium (25-50)", "High (>50)"]))

# 7. Probability Threshold Curve on Test Set (Controlled Comparison)
print("\n================================================================")
print("             PROBABILITY THRESHOLD SENSITIVITY CURVE            ")
print("================================================================")
thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'FNR (%)':<10} {'FPR (%)':<10} {'FP Count':<10} {'FN Count':<10}")
print("-" * 80)
for t in thresholds:
    t_pred = (y_prob >= t).astype(int)
    t_prec = precision_score(y_test, t_pred, zero_division=0)
    t_rec = recall_score(y_test, t_pred, zero_division=0)
    t_f1 = f1_score(y_test, t_pred, zero_division=0)
    t_cm = confusion_matrix(y_test, t_pred)
    t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
    t_fnr = (t_fn / (t_fn + t_tp)) * 100
    t_fpr = (t_fp / (t_fp + t_tn)) * 100
    print(f"{t:<10.2f} {t_prec:<10.4f} {t_rec:<10.4f} {t_f1:<10.4f} {t_fnr:<10.2f} {t_fpr:<10.2f} {t_fp:<10} {t_fn:<10}")

# 8. Check dataset generation physics in datasetc.py vs. real AEB physics
print("\n================================================================")
print("            DATASET GENERATION LOGIC AUDIT                     ")
print("================================================================")
import datasetc
print("simulate_collision definition in datasetc.py:")
lines, start = inspect.getsourcelines(datasetc.simulate_collision)
for l in lines:
    print("  " + l.rstrip())

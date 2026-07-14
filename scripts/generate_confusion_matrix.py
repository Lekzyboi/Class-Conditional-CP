"""
Generate confusion matrix artifacts from the raw predictions saved by
regenerate_test_predictions.py.

Reads:  assets/output/test_evaluation/raw/{test_softmax,test_argmax,test_labels}.npy
Writes: assets/output/test_evaluation/
  confusion_matrix.csv             8?8 raw counts + row/col totals
  confusion_matrix_normalized.csv  row-normalised (each row sums to 1.0)
  confusion_matrix.png             heatmap matching existing visual style
  prediction_frequencies.json      diagonal_pct and column_sum_pct per class
  confusion_summary.json           accuracy, per-class metrics, top-5 confusions

If any Task-3 verification fails the script saves to a staging directory
instead and exits with code 1 ? existing files are never overwritten on failure.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DIR     = "assets/output/test_evaluation/raw"
OUT_DIR     = "assets/output/test_evaluation"
STAGING_DIR = "assets/output/test_evaluation/staging_verify"
N_TOTAL     = 3096
N_CLASSES   = 8
CLASS_CODES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# Audit ground-truth values (from complete_results.json + prior audit)
AUDIT_CLASS_COUNTS = {
    "AK": 168, "BCC": 491, "BKL": 316, "DF": 46,
    "MEL": 679, "NV": 1272, "SCC": 87, "VASC": 37,
}
AUDIT_DIAG_SUM = 2205  # tolerance ?2

# confusion_analysis block (Standard CP ?=0.10 FAILED-coverage subset)
# Each entry: (true_class, pred_class, audit_count) ? new matrix must be ? audit_count
AUDIT_SUBSETS = [
    ("AK",  "BCC", 31), ("AK",  "BKL", 12), ("AK",  "MEL",  8),
    ("AK",  "NV",   3), ("AK",  "SCC",  2),
    ("BCC", "NV",  11), ("BCC", "BKL",  6), ("BCC", "MEL",  8),
    ("BCC", "AK",   1), ("BCC", "SCC",  2),
    ("BKL", "MEL", 25), ("BKL", "NV",  22), ("BKL", "BCC", 15),
    ("BKL", "AK",  13), ("BKL", "SCC",  4),
    ("DF",  "NV",   5), ("DF",  "BCC",  1), ("DF",  "MEL",  1),
    ("MEL", "NV",  47), ("MEL", "BCC", 10), ("MEL", "BKL",  4),
    ("MEL", "AK",   3), ("MEL", "SCC",  1),
    ("NV",  "MEL", 16), ("NV",  "BCC",  9), ("NV",  "BKL",  9),
    ("NV",  "AK",   8), ("NV",  "DF",   1),
    ("SCC", "BCC", 13), ("SCC", "BKL",  6), ("SCC", "AK",   3),
    ("SCC", "NV",   1), ("SCC", "MEL",  1),
    ("VASC","MEL",  4), ("VASC","BCC",  3), ("VASC","NV",   2),
]

# Manuscript "model predicted X in Y% of cases" claims to resolve
MANUSCRIPT_CLAIMS = {"NV": 35.5, "BCC": 15.3, "MEL": 15.1}

# ============================================================
# LOAD RAW PREDICTIONS
# ============================================================
def load_arrays():
    softmax = np.load(os.path.join(RAW_DIR, "test_softmax.npy"))
    argmax  = np.load(os.path.join(RAW_DIR, "test_argmax.npy"))
    labels  = np.load(os.path.join(RAW_DIR, "test_labels.npy"))
    assert softmax.shape == (N_TOTAL, N_CLASSES), \
        f"Unexpected softmax shape {softmax.shape}"
    assert argmax.shape  == (N_TOTAL,), f"Unexpected argmax shape {argmax.shape}"
    assert labels.shape  == (N_TOTAL,), f"Unexpected labels shape {labels.shape}"
    return softmax, argmax, labels

print("Loading raw prediction arrays ?")
softmax_arr, argmax_arr, label_arr = load_arrays()
print(f"  test_softmax : {softmax_arr.shape}  {softmax_arr.dtype}")
print(f"  test_argmax  : {argmax_arr.shape}   {argmax_arr.dtype}")
print(f"  test_labels  : {label_arr.shape}    {label_arr.dtype}")

# ============================================================
# BUILD CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(label_arr, argmax_arr, labels=list(range(N_CLASSES)))
# cm[i, j] = number of samples with true class i predicted as class j

code_idx = {c: i for i, c in enumerate(CLASS_CODES)}

# ============================================================
# TASK-3 VERIFICATIONS
# ============================================================
print("\n" + "=" * 62)
print("TASK-3 VERIFICATIONS")
print("=" * 62)

failures = []

# V1: diagonal sum
diag_sum = int(np.trace(cm))
v1 = abs(diag_sum - AUDIT_DIAG_SUM) <= 2
print(f"\nV1  Diagonal sum: {diag_sum}  (expected {AUDIT_DIAG_SUM} ?2)  {'OK' if v1 else 'FAIL'}")
if not v1:
    failures.append(f"V1 diagonal sum {diag_sum} != {AUDIT_DIAG_SUM} ?2")

# V2: per-class row sums
print("\nV2  Per-class row sums:")
v2 = True
for i, code in enumerate(CLASS_CODES):
    actual   = int(cm[i].sum())
    expected = AUDIT_CLASS_COUNTS[code]
    ok = actual == expected
    if not ok:
        v2 = False
        failures.append(f"V2 row sum {code}: {actual} != {expected}")
    print(f"  {code:5s}: {actual:5d}  expected {expected:5d}  {'OK' if ok else 'FAIL'}")

# V3: confusion_analysis is a strict subset
print("\nV3  confusion_analysis ? new matrix off-diagonal:")
v3 = True
subset_detail = []
for true_cls, pred_cls, audit_count in AUDIT_SUBSETS:
    ti  = code_idx[true_cls]
    pi  = code_idx[pred_cls]
    new = int(cm[ti, pi])
    ok  = new >= audit_count
    if not ok:
        v3 = False
        failures.append(f"V3 cm[{true_cls}->{pred_cls}]={new} < audit {audit_count}")
    subset_detail.append((true_cls, pred_cls, audit_count, new, ok))
    print(f"  [{true_cls:4s}->{pred_cls:4s}]  new={new:3d}  audit?{audit_count:2d}  {'OK' if ok else 'FAIL'}")

all_pass = v1 and v2 and v3
save_dir = OUT_DIR if all_pass else STAGING_DIR
os.makedirs(save_dir, exist_ok=True)

if not all_pass:
    print("\n!  Verification failures detected ? saving to STAGING, not overwriting existing files.")
    for f in failures:
        print(f"   FAIL {f}")
else:
    print("\nAll verifications PASSED ? saving to final output directory.")

# ============================================================
# OUTPUT 1: Raw counts CSV (rows = true, cols = predicted)
# ============================================================
cm_df = pd.DataFrame(cm, index=CLASS_CODES, columns=CLASS_CODES)
cm_df.index.name = "true \\ pred"
cm_df["ROW_TOTAL"] = cm_df.sum(axis=1)
col_totals = pd.DataFrame(
    [cm_df.sum(axis=0).values],
    index=["COL_TOTAL"],
    columns=cm_df.columns,
)
cm_df_full = pd.concat([cm_df, col_totals])

csv_path = os.path.join(save_dir, "confusion_matrix.csv")
cm_df_full.to_csv(csv_path)
print(f"\nSaved: {csv_path}")

# ============================================================
# OUTPUT 2: Row-normalised CSV
# ============================================================
cm_float = cm.astype(float)
row_sums  = cm_float.sum(axis=1, keepdims=True)
cm_norm   = cm_float / np.where(row_sums == 0, 1.0, row_sums)
norm_df   = pd.DataFrame(
    np.round(cm_norm, 4), index=CLASS_CODES, columns=CLASS_CODES
)
norm_df.index.name = "true \\ pred"
norm_csv  = os.path.join(save_dir, "confusion_matrix_normalized.csv")
norm_df.to_csv(norm_csv)
print(f"Saved: {norm_csv}")

# ============================================================
# OUTPUT 3: Heatmap PNG ? matches existing visual style
#   (Blues colormap, black bold labels on axes, white annotations,
#    seaborn linewidths, title with accuracy)
# ============================================================
overall_acc = float(np.mean(argmax_arr == label_arr))

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_CODES,
    yticklabels=CLASS_CODES,
    ax=ax,
    linewidths=0.5,
    linecolor="gray",
    annot_kws={"size": 11},
    cbar_kws={"label": "Count", "shrink": 0.82},
)
ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", labelpad=10)
ax.set_ylabel("True Label",      fontsize=13, fontweight="bold", labelpad=10)
ax.set_title(
    f"Confusion Matrix - ISIC 2019 Test Set\n"
    f"Overall Accuracy: {overall_acc:.4f}  ({diag_sum}/{N_TOTAL})",
    fontsize=13, fontweight="bold", pad=14,
)
ax.tick_params(axis="both", labelsize=11)
plt.tight_layout()

png_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {png_path}")

# ============================================================
# OUTPUT 4: prediction_frequencies.json
# ============================================================
pred_freq = {}
for i, code in enumerate(CLASS_CODES):
    diag_n   = int(cm[i, i])
    col_n    = int(cm[:, i].sum())
    pred_freq[code] = {
        "diagonal_count":  diag_n,
        "diagonal_pct":    round(100.0 * diag_n / N_TOTAL, 2),
        "column_sum_count": col_n,
        "column_sum_pct":  round(100.0 * col_n / N_TOTAL, 2),
    }

pf_path = os.path.join(save_dir, "prediction_frequencies.json")
with open(pf_path, "w") as f:
    json.dump(pred_freq, f, indent=2)
print(f"Saved: {pf_path}")

# ============================================================
# OUTPUT 5: confusion_summary.json
# ============================================================
precision = precision_score(
    label_arr, argmax_arr, average=None,
    labels=list(range(N_CLASSES)), zero_division=0,
)
recall = recall_score(
    label_arr, argmax_arr, average=None,
    labels=list(range(N_CLASSES)), zero_division=0,
)
f1 = f1_score(
    label_arr, argmax_arr, average=None,
    labels=list(range(N_CLASSES)), zero_division=0,
)

per_class_metrics = {}
for i, code in enumerate(CLASS_CODES):
    per_class_metrics[code] = {
        "n":         AUDIT_CLASS_COUNTS[code],
        "correct":   int(cm[i, i]),
        "precision": round(float(precision[i]), 4),
        "recall":    round(float(recall[i]), 4),
        "f1":        round(float(f1[i]), 4),
    }

# Top-10 off-diagonal pairs (by raw count)
off_diag_pairs = []
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        if i != j and cm[i, j] > 0:
            off_diag_pairs.append({
                "true":          CLASS_CODES[i],
                "pred":          CLASS_CODES[j],
                "count":         int(cm[i, j]),
                "pct_of_true":   round(100.0 * cm[i, j] / cm[i].sum(), 1),
                "pct_of_total":  round(100.0 * cm[i, j] / N_TOTAL, 2),
            })
off_diag_pairs.sort(key=lambda x: x["count"], reverse=True)
top5 = off_diag_pairs[:5]

summary = {
    "overall_accuracy": round(overall_acc, 4),
    "diagonal_sum":     diag_sum,
    "n_total":          N_TOTAL,
    "per_class_metrics": per_class_metrics,
    "column_sums":      {
        code: int(cm[:, i].sum()) for i, code in enumerate(CLASS_CODES)
    },
    "top10_confused_pairs": off_diag_pairs[:10],
    "verification": {
        "V1_diagonal_sum": "OK" if v1 else "FAIL",
        "V2_row_sums":     "OK" if v2 else "FAIL",
        "V3_subset_check": "OK" if v3 else "FAIL",
        "all_pass":        all_pass,
        "save_directory":  save_dir,
    },
}

summ_path = os.path.join(save_dir, "confusion_summary.json")
with open(summ_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summ_path}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 62)
print("FINAL REPORT")
print("=" * 62)

# Files
print("\nFiles created:")
for fname in [
    "confusion_matrix.csv", "confusion_matrix_normalized.csv",
    "confusion_matrix.png", "prediction_frequencies.json",
    "confusion_summary.json",
]:
    p = os.path.join(save_dir, fname)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {p}  ({size:,} bytes)")

# Verification summary
print(f"\nVerification results:")
print(f"  V1 Diagonal sum = {diag_sum} (expected {AUDIT_DIAG_SUM} ?2): {'OK' if v1 else 'FAIL'}")
print(f"  V2 Per-class row sums match audit:                           {'OK' if v2 else 'FAIL'}")
print(f"  V3 confusion_analysis is subset of off-diagonal cells:       {'OK' if v3 else 'FAIL'}")

# Prediction frequency table
print(f"\nPrediction frequencies (N={N_TOTAL}):")
print(f"  {'Class':6s}  {'diag_n':>7s}  {'diag_pct':>9s}  {'col_n':>7s}  {'col_pct':>8s}")
print(f"  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*8}")
for code in CLASS_CODES:
    pf = pred_freq[code]
    print(f"  {code:6s}  {pf['diagonal_count']:>7d}  "
          f"{pf['diagonal_pct']:>8.2f}%  "
          f"{pf['column_sum_count']:>7d}  "
          f"{pf['column_sum_pct']:>7.2f}%")

# Resolve manuscript claims
print(f"\nManuscript ? 3.4 claim resolution ('model predicted X in Y% of cases'):")
for code, claimed_pct in MANUSCRIPT_CLAIMS.items():
    pf = pred_freq[code]
    d_diff = abs(pf["diagonal_pct"]  - claimed_pct)
    c_diff = abs(pf["column_sum_pct"] - claimed_pct)
    if d_diff <= 0.15:
        match = f"diagonal_pct ({pf['diagonal_pct']}%)  [diff {d_diff:.2f}pp]"
    elif c_diff <= 0.15:
        match = f"column_sum_pct ({pf['column_sum_pct']}%)  [diff {c_diff:.2f}pp]"
    else:
        match = (f"NEITHER  (diagonal={pf['diagonal_pct']}% diff={d_diff:.2f}pp, "
                 f"col_sum={pf['column_sum_pct']}% diff={c_diff:.2f}pp)")
    print(f"  {code} {claimed_pct}% -> matches {match}")

# Top-5 confused pairs
print(f"\nTop 5 most-confused class pairs:")
for rank, pair in enumerate(top5, 1):
    print(f"  #{rank}  true={pair['true']:4s} -> pred={pair['pred']:4s}  "
          f"n={pair['count']:3d}  "
          f"({pair['pct_of_true']}% of true {pair['true']},  "
          f"{pair['pct_of_total']}% of all)")

# BKL->MEL check
bkl_mel = next(
    (p for p in off_diag_pairs if p["true"] == "BKL" and p["pred"] == "MEL"), None
)
bkl_mel_rank = off_diag_pairs.index(bkl_mel) + 1 if bkl_mel else None
print(f"\nBKL->MEL (manuscript: 'clinically concerning benign-malignant confusion'):")
if bkl_mel:
    in_top5 = bkl_mel_rank <= 5
    print(f"  count={bkl_mel['count']}  rank=#{bkl_mel_rank}  "
          f"{'IS in top 5 OK' if in_top5 else 'NOT in top 5'}")
    print(f"  {bkl_mel['pct_of_true']}% of true BKL cases misclassified as MEL")
    print(f"  Verdict: {'CONFIRMED as top confusion' if in_top5 else 'Present but not top-5'}")

if not all_pass:
    print(f"\n!  Outputs saved to STAGING ({STAGING_DIR})  ? existing files unchanged.")
    sys.exit(1)

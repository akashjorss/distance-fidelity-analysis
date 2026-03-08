"""Binary classification: predict PSNR >= 25 dB.

XGBoost classifier, 5-fold CV. Reports AUC-ROC, precision, recall, F1.

Usage:
    python binary_classification.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import ALL_METRICS, get_metric_label, savefig, to_latex_table, FIGURES_DIR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def main():
    parser = argparse.ArgumentParser(description="Binary classification analysis")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=25.0, help="PSNR threshold (dB)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    available_metrics = [m for m in ALL_METRICS if m in df.columns and df[m].notna().sum() > 100]
    metadata_cols = []
    if "budget" in df.columns:
        metadata_cols.append("budget")
    if "scene" in df.columns:
        df["scene_id"] = df["scene"].astype("category").cat.codes
        metadata_cols.append("scene_id")

    feature_cols = available_metrics + metadata_cols
    subset = df[feature_cols + ["psnr"]].dropna()

    # Create binary target
    y = (subset["psnr"] >= args.threshold).astype(int).values
    X = subset[feature_cols].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\nThreshold: {args.threshold} dB")
    print(f"  Positive (>= {args.threshold}): {n_pos} ({100*n_pos/len(y):.1f}%)")
    print(f"  Negative (< {args.threshold}):  {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  Features: {len(feature_cols)}")

    # K-fold CV
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    metrics_per_fold = {"auc": [], "precision": [], "recall": [], "f1": []}
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Handle class imbalance
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=args.seed + fold, n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        metrics_per_fold["auc"].append(auc)
        metrics_per_fold["precision"].append(prec)
        metrics_per_fold["recall"].append(rec)
        metrics_per_fold["f1"].append(f1)

        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold+1}: AUC={auc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    # Summary
    print(f"\n{'='*60}")
    for metric_name in ["auc", "precision", "recall", "f1"]:
        vals = metrics_per_fold[metric_name]
        print(f"  {metric_name.upper():10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # ROC curve
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    if len(np.unique(all_y_true)) > 1:
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        overall_auc = roc_auc_score(all_y_true, all_y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {overall_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: PSNR >= {args.threshold} dB")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        savefig(fig, "binary_classification_roc")

    # Feature importance
    model_full = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed, n_jobs=-1,
    )
    model_full.fit(X, y, verbose=False)
    importances = model_full.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.3)))
    labels = [get_metric_label(f) if f in ALL_METRICS else f for f in feature_cols]
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#2ca02c")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([labels[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Binary Classification: PSNR >= {args.threshold} dB")
    savefig(fig, "binary_classification_importance")


if __name__ == "__main__":
    main()

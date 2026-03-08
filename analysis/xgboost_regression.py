"""XGBoost regression: predict fidelity from distance metrics.

5-fold cross-validation. Compare:
  - All 12 metrics
  - Original 7 only (CCIS baseline)
  - New 5 only (InfoMax3D, Plucker, Angular, Euclidean, LPIPS)

Reports R^2, MAE, Spearman rho.

Usage:
    python xgboost_regression.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import (
    ALL_METRICS, ORIGINAL_METRICS, NEW_METRICS,
    get_metric_label, savefig, to_latex_table, FIGURES_DIR,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_and_evaluate(X, y, feature_names, n_folds=5, seed=42):
    """Train XGBoost with K-fold CV, return metrics and best model."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    r2_scores = []
    mae_scores = []
    rho_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed + fold,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rho, _ = spearmanr(y_val, y_pred)

        r2_scores.append(r2)
        mae_scores.append(mae)
        rho_scores.append(rho)
        models.append(model)

    best_idx = np.argmax(r2_scores)
    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rho_mean": np.mean(rho_scores),
        "rho_std": np.std(rho_scores),
        "best_model": models[best_idx],
        "feature_names": feature_names,
        "per_fold": {
            "r2": r2_scores,
            "mae": mae_scores,
            "rho": rho_scores,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="XGBoost regression for fidelity prediction")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="psnr", choices=["psnr", "ssim", "lpips"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    # Define feature sets
    available_metrics = [m for m in ALL_METRICS if m in df.columns and df[m].notna().sum() > 100]
    available_original = [m for m in ORIGINAL_METRICS if m in available_metrics]
    available_new = [m for m in NEW_METRICS if m in available_metrics]

    feature_sets = {
        "All metrics": available_metrics,
        "Original 7": available_original,
        "New 5": available_new,
    }

    # Add metadata features
    metadata_cols = []
    if "budget" in df.columns:
        metadata_cols.append("budget")

    # Encode scene as numeric
    if "scene" in df.columns:
        df["scene_id"] = df["scene"].astype("category").cat.codes
        metadata_cols.append("scene_id")

    print(f"\nTarget: {args.target.upper()}")
    print(f"Available metrics: {len(available_metrics)}")
    print(f"  Original: {available_original}")
    print(f"  New: {available_new}")

    results_summary = {}

    for set_name, metric_cols in feature_sets.items():
        if not metric_cols:
            print(f"\n  {set_name}: No metrics available, skipping")
            continue

        feature_cols = metric_cols + metadata_cols
        subset = df[feature_cols + [args.target]].dropna()
        if len(subset) < 50:
            print(f"\n  {set_name}: Only {len(subset)} complete rows, skipping")
            continue

        X = subset[feature_cols].values
        y = subset[args.target].values

        print(f"\n{'='*60}")
        print(f"  {set_name} ({len(metric_cols)} metrics + {len(metadata_cols)} metadata)")
        print(f"  Samples: {len(subset)}")

        result = train_and_evaluate(X, y, feature_cols, n_folds=args.folds, seed=args.seed)
        results_summary[set_name] = result

        print(f"  R²:  {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")
        print(f"  MAE: {result['mae_mean']:.4f} ± {result['mae_std']:.4f}")
        print(f"  ρ:   {result['rho_mean']:.4f} ± {result['rho_std']:.4f}")

        # Feature importance
        model = result["best_model"]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"  Top features:")
        for i in sorted_idx[:5]:
            label = get_metric_label(feature_cols[i]) if feature_cols[i] in ALL_METRICS else feature_cols[i]
            print(f"    {label}: {importances[i]:.4f}")

    # ── Comparison bar chart ──
    if results_summary:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        set_names = list(results_summary.keys())
        x = range(len(set_names))

        for ax_idx, (metric_name, metric_key) in enumerate([("R²", "r2"), ("MAE", "mae"), ("ρ (Spearman)", "rho")]):
            means = [results_summary[s][f"{metric_key}_mean"] for s in set_names]
            stds = [results_summary[s][f"{metric_key}_std"] for s in set_names]
            axes[ax_idx].bar(x, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(set_names)])
            axes[ax_idx].set_xticks(x)
            axes[ax_idx].set_xticklabels(set_names, rotation=15)
            axes[ax_idx].set_ylabel(metric_name)
            axes[ax_idx].set_title(f"{metric_name} ({args.folds}-fold CV)")

        plt.suptitle(f"XGBoost {args.target.upper()} Prediction")
        plt.tight_layout()
        savefig(fig, f"xgboost_{args.target}_comparison")

    # ── Feature importance plot (all metrics) ──
    if "All metrics" in results_summary:
        result = results_summary["All metrics"]
        model = result["best_model"]
        feature_cols = result["feature_names"]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.3)))
        labels = [get_metric_label(feature_cols[i]) if feature_cols[i] in ALL_METRICS else feature_cols[i]
                  for i in sorted_idx]
        ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#1f77b4")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(f"XGBoost Feature Importance ({args.target.upper()})")
        savefig(fig, f"xgboost_{args.target}_importance")

    # ── Save best model ──
    if "All metrics" in results_summary:
        model_path = os.path.join(FIGURES_DIR, f"xgboost_{args.target}_best.json")
        results_summary["All metrics"]["best_model"].save_model(model_path)
        print(f"\nBest model saved to {model_path}")

    # ── LaTeX summary table ──
    if results_summary:
        rows = []
        for name, res in results_summary.items():
            rows.append({
                "Feature Set": name,
                "$R^2$": f"{res['r2_mean']:.3f} \\pm {res['r2_std']:.3f}",
                "MAE": f"{res['mae_mean']:.3f} \\pm {res['mae_std']:.3f}",
                "$\\rho$": f"{res['rho_mean']:.3f} \\pm {res['rho_std']:.3f}",
            })
        latex_df = pd.DataFrame(rows).set_index("Feature Set")
        latex_str = to_latex_table(
            latex_df,
            caption=f"XGBoost {args.target.upper()} prediction ({args.folds}-fold CV)",
            label=f"xgboost_{args.target}",
            bold_best=False,
        )
        latex_path = os.path.join(FIGURES_DIR, f"xgboost_{args.target}_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()

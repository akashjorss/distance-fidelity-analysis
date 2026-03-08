"""SHAP analysis of best XGBoost model.

Produces beeswarm and bar plots. Key question: do Plucker/InfoMax3D dominate?

Usage:
    python shap_analysis.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import ALL_METRICS, get_metric_label, savefig, FIGURES_DIR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="psnr", choices=["psnr", "ssim", "lpips"])
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples for SHAP")
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
    subset = df[feature_cols + [args.target]].dropna()
    print(f"Complete rows: {len(subset)}")

    X = subset[feature_cols].values
    y = subset[args.target].values

    # Train full model
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y, verbose=False)

    # SHAP values
    n_samples = min(args.max_samples, len(X))
    idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
    X_shap = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Rename features for display
    display_names = [get_metric_label(f) if f in ALL_METRICS else f for f in feature_cols]

    # ── Beeswarm plot ──
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.35)))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=display_names,
        show=False,
        max_display=len(feature_cols),
    )
    plt.title(f"SHAP Values for {args.target.upper()} Prediction")
    plt.tight_layout()
    savefig(plt.gcf(), f"shap_beeswarm_{args.target}")

    # ── Bar plot (mean |SHAP|) ──
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.3)))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=display_names,
        plot_type="bar",
        show=False,
        max_display=len(feature_cols),
    )
    plt.title(f"Mean |SHAP| for {args.target.upper()} Prediction")
    plt.tight_layout()
    savefig(plt.gcf(), f"shap_bar_{args.target}")

    # ── Print summary ──
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    print(f"\n=== SHAP Feature Importance ({args.target.upper()}) ===")
    for i in sorted_idx:
        print(f"  {display_names[i]:20s}: {mean_abs_shap[i]:.4f}")

    # ── Geometric vs Visual comparison ──
    from plot_utils import GEOMETRIC_METRICS, VISUAL_METRICS
    geo_shap = sum(mean_abs_shap[feature_cols.index(m)] for m in GEOMETRIC_METRICS if m in feature_cols)
    vis_shap = sum(mean_abs_shap[feature_cols.index(m)] for m in VISUAL_METRICS if m in feature_cols)
    total = mean_abs_shap.sum()

    print(f"\n  Geometric metrics SHAP: {geo_shap:.4f} ({100*geo_shap/total:.1f}%)")
    print(f"  Visual metrics SHAP:    {vis_shap:.4f} ({100*vis_shap/total:.1f}%)")
    print(f"  Other:                  {total-geo_shap-vis_shap:.4f} ({100*(total-geo_shap-vis_shap)/total:.1f}%)")


if __name__ == "__main__":
    main()

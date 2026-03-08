"""Cross-dataset generalization: train on one dataset, test on another.

Key new analysis: tests whether metric importance transfers across capture styles.
E.g., train on LLFF -> test on T&T (and all other pairs).

Usage:
    python cross_dataset_generalization.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb
import shap

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import ALL_METRICS, GEOMETRIC_METRICS, VISUAL_METRICS, get_metric_label, savefig, FIGURES_DIR, DATASET_LABELS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset generalization")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="psnr", choices=["psnr", "ssim", "lpips"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    available_metrics = [m for m in ALL_METRICS if m in df.columns and df[m].notna().sum() > 100]
    metadata_cols = ["budget"] if "budget" in df.columns else []
    feature_cols = available_metrics + metadata_cols

    subset = df[feature_cols + [args.target, "dataset"]].dropna()
    datasets = sorted(subset["dataset"].unique())
    print(f"Datasets: {datasets}, Features: {len(feature_cols)}")

    # ── Pairwise cross-dataset evaluation ──
    results_matrix = np.zeros((len(datasets), len(datasets)))
    rho_matrix = np.zeros((len(datasets), len(datasets)))
    importance_per_pair = {}

    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            train_data = subset[subset["dataset"] == train_ds]
            test_data = subset[subset["dataset"] == test_ds]

            if len(train_data) < 20 or len(test_data) < 10:
                results_matrix[i, j] = np.nan
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data[args.target].values
            X_test = test_data[feature_cols].values
            y_test = test_data[args.target].values

            model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rho, _ = spearmanr(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            results_matrix[i, j] = r2
            rho_matrix[i, j] = rho

            train_label = DATASET_LABELS.get(train_ds, train_ds)
            test_label = DATASET_LABELS.get(test_ds, test_ds)
            marker = " (in-domain)" if train_ds == test_ds else ""
            print(f"  {train_label} -> {test_label}: R²={r2:.4f}  ρ={rho:.4f}  MAE={mae:.4f}{marker}")

            # Store feature importance
            importance_per_pair[(train_ds, test_ds)] = model.feature_importances_

    # ── R² heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ds_labels = [DATASET_LABELS.get(d, d) for d in datasets]

    for ax, matrix, title in [(axes[0], results_matrix, "R²"), (axes[1], rho_matrix, "Spearman ρ")]:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=1, aspect="auto")
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(ds_labels)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(ds_labels)
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")
        ax.set_title(f"Cross-Dataset {title}")

        for ii in range(len(datasets)):
            for jj in range(len(datasets)):
                val = matrix[ii, jj]
                if not np.isnan(val):
                    color = "white" if val < 0.3 else "black"
                    weight = "bold" if ii == jj else "normal"
                    ax.text(jj, ii, f"{val:.3f}", ha="center", va="center",
                            color=color, fontweight=weight, fontsize=11)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Cross-Dataset Generalization ({args.target.upper()})")
    plt.tight_layout()
    savefig(fig, f"cross_dataset_{args.target}")

    # ── Feature importance comparison across transfer pairs ──
    if importance_per_pair:
        # Compare which features transfer well
        fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.3)))

        # Aggregate: in-domain vs cross-domain feature importance
        in_domain_imp = np.zeros(len(feature_cols))
        cross_domain_imp = np.zeros(len(feature_cols))
        n_in = 0
        n_cross = 0

        for (train_ds, test_ds), imp in importance_per_pair.items():
            if train_ds == test_ds:
                in_domain_imp += imp
                n_in += 1
            else:
                cross_domain_imp += imp
                n_cross += 1

        if n_in > 0:
            in_domain_imp /= n_in
        if n_cross > 0:
            cross_domain_imp /= n_cross

        sorted_idx = np.argsort(cross_domain_imp)
        labels = [get_metric_label(f) if f in ALL_METRICS else f for f in feature_cols]

        y_pos = np.arange(len(sorted_idx))
        height = 0.35

        ax.barh(y_pos - height/2, in_domain_imp[sorted_idx], height, label="In-domain", color="#1f77b4", alpha=0.8)
        ax.barh(y_pos + height/2, cross_domain_imp[sorted_idx], height, label="Cross-domain", color="#ff7f0e", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[i] for i in sorted_idx])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importance: In-domain vs Cross-domain ({args.target.upper()})")
        ax.legend()
        savefig(fig, f"cross_dataset_importance_{args.target}")

        # Print summary
        print(f"\n=== Feature Importance Transfer ===")
        print(f"{'Feature':25s} {'In-domain':>10s} {'Cross-domain':>12s} {'Ratio':>8s}")
        for i in np.argsort(cross_domain_imp)[::-1]:
            label = labels[i]
            ind = in_domain_imp[i]
            crd = cross_domain_imp[i]
            ratio = crd / (ind + 1e-8)
            print(f"  {label:23s} {ind:10.4f} {crd:12.4f} {ratio:8.2f}")

        # Geometric vs Visual transfer
        geo_cross = sum(cross_domain_imp[feature_cols.index(m)] for m in GEOMETRIC_METRICS if m in feature_cols)
        vis_cross = sum(cross_domain_imp[feature_cols.index(m)] for m in VISUAL_METRICS if m in feature_cols)
        total_cross = cross_domain_imp.sum()

        print(f"\n  Cross-domain geometric share: {100*geo_cross/total_cross:.1f}%")
        print(f"  Cross-domain visual share:    {100*vis_cross/total_cross:.1f}%")


if __name__ == "__main__":
    main()

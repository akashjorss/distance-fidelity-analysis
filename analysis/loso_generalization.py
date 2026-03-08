"""Leave-One-Scene-Out (LOSO) generalization within each dataset.

For each dataset, trains on all scenes except one, tests on the held-out scene.
Reports per-scene R^2 and checks if more data (vs original CCIS) fixes the LOSO failure.

Usage:
    python loso_generalization.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import ALL_METRICS, get_metric_label, savefig, to_latex_table, FIGURES_DIR, DATASET_LABELS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="LOSO generalization analysis")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="psnr", choices=["psnr", "ssim", "lpips"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    available_metrics = [m for m in ALL_METRICS if m in df.columns and df[m].notna().sum() > 100]
    metadata_cols = ["budget"] if "budget" in df.columns else []

    feature_cols = available_metrics + metadata_cols
    subset = df[feature_cols + [args.target, "dataset", "scene"]].dropna()
    print(f"Complete rows: {len(subset)}")

    datasets = sorted(subset["dataset"].unique())
    all_results = {}

    for ds in datasets:
        ds_data = subset[subset["dataset"] == ds]
        scenes = sorted(ds_data["scene"].unique())
        ds_label = DATASET_LABELS.get(ds, ds)

        print(f"\n{'='*60}")
        print(f"  {ds_label} ({len(ds_data)} frames, {len(scenes)} scenes)")

        scene_results = {}
        for held_out in scenes:
            train_data = ds_data[ds_data["scene"] != held_out]
            test_data = ds_data[ds_data["scene"] == held_out]

            if len(train_data) < 20 or len(test_data) < 5:
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data[args.target].values
            X_test = test_data[feature_cols].values
            y_test = test_data[args.target].values

            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rho, _ = spearmanr(y_test, y_pred)

            scene_results[held_out] = {"r2": r2, "mae": mae, "rho": rho, "n": len(test_data)}
            print(f"    {held_out:15s}: R²={r2:+.4f}  MAE={mae:.4f}  ρ={rho:.4f}  (n={len(test_data)})")

        if scene_results:
            avg_r2 = np.mean([v["r2"] for v in scene_results.values()])
            avg_rho = np.mean([v["rho"] for v in scene_results.values()])
            n_positive_r2 = sum(1 for v in scene_results.values() if v["r2"] > 0)
            print(f"  --- {ds_label} Average: R²={avg_r2:.4f}  ρ={avg_rho:.4f}  "
                  f"({n_positive_r2}/{len(scene_results)} scenes with R²>0)")

        all_results[ds] = scene_results

    # ── Cross-dataset LOSO (all datasets pooled) ──
    print(f"\n{'='*60}")
    print("  All datasets pooled LOSO")

    all_scenes = sorted(subset["scene"].unique())
    pooled_results = {}

    for held_out in all_scenes:
        train_data = subset[subset["scene"] != held_out]
        test_data = subset[subset["scene"] == held_out]

        if len(train_data) < 20 or len(test_data) < 5:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[args.target].values
        X_test = test_data[feature_cols].values
        y_test = test_data[args.target].values

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rho, _ = spearmanr(y_test, y_pred)
        ds = test_data["dataset"].iloc[0]
        pooled_results[held_out] = {"r2": r2, "rho": rho, "dataset": ds, "n": len(test_data)}

    if pooled_results:
        avg_r2 = np.mean([v["r2"] for v in pooled_results.values()])
        n_pos = sum(1 for v in pooled_results.values() if v["r2"] > 0)
        print(f"  Average R²: {avg_r2:.4f}  ({n_pos}/{len(pooled_results)} with R²>0)")

    # ── Visualization ──
    fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(4 * (len(datasets) + 1), 5), sharey=True)
    if len(datasets) + 1 == 1:
        axes = [axes]

    for ax_idx, ds in enumerate(datasets):
        if ds not in all_results or not all_results[ds]:
            continue
        scene_results = all_results[ds]
        scenes = sorted(scene_results.keys())
        r2_vals = [scene_results[s]["r2"] for s in scenes]
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in r2_vals]

        axes[ax_idx].barh(range(len(scenes)), r2_vals, color=colors)
        axes[ax_idx].set_yticks(range(len(scenes)))
        axes[ax_idx].set_yticklabels(scenes)
        axes[ax_idx].axvline(x=0, color="k", lw=0.5)
        axes[ax_idx].set_xlabel("R²")
        axes[ax_idx].set_title(DATASET_LABELS.get(ds, ds))

    # Pooled plot
    if pooled_results:
        scenes = sorted(pooled_results.keys())
        r2_vals = [pooled_results[s]["r2"] for s in scenes]
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in r2_vals]
        axes[-1].barh(range(len(scenes)), r2_vals, color=colors)
        axes[-1].set_yticks(range(len(scenes)))
        axes[-1].set_yticklabels(scenes, fontsize=7)
        axes[-1].axvline(x=0, color="k", lw=0.5)
        axes[-1].set_xlabel("R²")
        axes[-1].set_title("All Pooled")

    plt.suptitle(f"LOSO Generalization ({args.target.upper()})")
    plt.tight_layout()
    savefig(fig, f"loso_{args.target}")

    # ── LaTeX table ──
    rows = []
    for ds in datasets:
        if ds not in all_results:
            continue
        sr = all_results[ds]
        for scene in sorted(sr.keys()):
            rows.append({
                "Dataset": DATASET_LABELS.get(ds, ds),
                "Scene": scene,
                "$R^2$": sr[scene]["r2"],
                "$\\rho$": sr[scene]["rho"],
                "n": sr[scene]["n"],
            })
    if rows:
        latex_df = pd.DataFrame(rows)
        latex_path = os.path.join(FIGURES_DIR, f"loso_{args.target}_table.tex")
        latex_df.to_latex(latex_path, index=False, float_format="%.3f", escape=False)
        print(f"\nLaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()

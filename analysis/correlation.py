"""Spearman/Pearson correlation of each distance metric vs fidelity metrics.

Produces:
  - Heatmaps of correlation coefficients
  - LaTeX tables
  - Stratified analysis by dataset

Usage:
    python correlation.py --csv ../results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import (
    ALL_METRICS, METRIC_LABELS, DATASET_LABELS,
    get_metric_label, correlation_heatmap, savefig, to_latex_table, FIGURES_DIR,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_correlations(df, distance_cols, fidelity_cols, method="spearman"):
    """Compute correlation between each distance and fidelity metric."""
    results = {}
    for fid_col in fidelity_cols:
        results[fid_col] = {}
        for dist_col in distance_cols:
            valid = df[[dist_col, fid_col]].dropna()
            if len(valid) < 10:
                results[fid_col][dist_col] = {"rho": np.nan, "p": np.nan, "n": len(valid)}
                continue

            if method == "spearman":
                rho, p = stats.spearmanr(valid[dist_col], valid[fid_col])
            else:
                rho, p = stats.pearsonr(valid[dist_col], valid[fid_col])

            results[fid_col][dist_col] = {"rho": rho, "p": p, "n": len(valid)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument("--csv", type=str, required=True, help="Path to combined_perframe.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Identify available distance and fidelity columns
    fidelity_cols = [c for c in ["psnr", "ssim", "lpips"] if c in df.columns]
    distance_cols = [c for c in ALL_METRICS if c in df.columns and df[c].notna().sum() > 10]

    print(f"Fidelity columns: {fidelity_cols}")
    print(f"Distance columns ({len(distance_cols)}): {distance_cols}")

    # ── 1. Overall correlation ──
    print("\n=== Overall Spearman Correlation ===")
    corr_spearman = compute_correlations(df, distance_cols, fidelity_cols, method="spearman")
    corr_pearson = compute_correlations(df, distance_cols, fidelity_cols, method="pearson")

    # Print table
    for fid in fidelity_cols:
        print(f"\n  vs {fid.upper()}:")
        rows = []
        for dist in distance_cols:
            s = corr_spearman[fid][dist]
            p = corr_pearson[fid][dist]
            print(f"    {get_metric_label(dist):20s}: Spearman={s['rho']:+.4f} (p={s['p']:.2e})  "
                  f"Pearson={p['rho']:+.4f} (p={p['p']:.2e})  n={s['n']}")
            rows.append({
                "Metric": get_metric_label(dist),
                "Spearman": s["rho"],
                "Pearson": p["rho"],
            })

        # Create heatmap
        rho_df = pd.DataFrame({
            get_metric_label(d): corr_spearman[fid][d]["rho"]
            for d in distance_cols
        }, index=[fid.upper()])
        corr_matrix = pd.DataFrame(
            [[corr_spearman[fid][d]["rho"] for d in distance_cols]],
            columns=[get_metric_label(d) for d in distance_cols],
            index=[fid.upper()],
        )

    # Full correlation heatmap (distance metrics vs PSNR)
    if "psnr" in corr_spearman:
        corr_vals = pd.DataFrame(
            {get_metric_label(d): [corr_spearman[fid][d]["rho"] for fid in fidelity_cols]
             for d in distance_cols},
            index=[f.upper() for f in fidelity_cols],
        )
        fig, ax = plt.subplots(figsize=(max(8, len(distance_cols) * 0.7), 3))
        im = ax.imshow(corr_vals.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(distance_cols)))
        ax.set_xticklabels(corr_vals.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(fidelity_cols)))
        ax.set_yticklabels(corr_vals.index)
        for i in range(len(fidelity_cols)):
            for j in range(len(distance_cols)):
                val = corr_vals.values[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Spearman Correlation: Distance Metrics vs Fidelity (All Data)")
        savefig(fig, "correlation_overall")

    # ── 2. Stratified by dataset ──
    print("\n=== Stratified by Dataset ===")
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(max(8, len(distance_cols) * 0.7), 3 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]

    for idx, ds in enumerate(sorted(datasets)):
        df_ds = df[df["dataset"] == ds]
        corr_ds = compute_correlations(df_ds, distance_cols, fidelity_cols, method="spearman")
        ds_label = DATASET_LABELS.get(ds, ds)

        print(f"\n  {ds_label} ({len(df_ds)} frames):")
        for fid in fidelity_cols:
            print(f"    vs {fid.upper()}:")
            for dist in distance_cols:
                s = corr_ds[fid][dist]
                print(f"      {get_metric_label(dist):20s}: rho={s['rho']:+.4f}  n={s['n']}")

        # Subplot
        if "psnr" in corr_ds:
            vals = np.array([[corr_ds[fid][d]["rho"] for d in distance_cols] for fid in fidelity_cols])
            im = axes[idx].imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            axes[idx].set_xticks(range(len(distance_cols)))
            axes[idx].set_xticklabels([get_metric_label(d) for d in distance_cols], rotation=45, ha="right", fontsize=7)
            axes[idx].set_yticks(range(len(fidelity_cols)))
            axes[idx].set_yticklabels([f.upper() for f in fidelity_cols])
            axes[idx].set_title(f"{ds_label} (n={len(df_ds)})")
            for i in range(len(fidelity_cols)):
                for j in range(len(distance_cols)):
                    if not np.isnan(vals[i, j]):
                        color = "white" if abs(vals[i, j]) > 0.5 else "black"
                        axes[idx].text(j, i, f"{vals[i,j]:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.suptitle("Spearman Correlation by Dataset")
    savefig(fig, "correlation_by_dataset")

    # ── 3. LaTeX output ──
    if "psnr" in corr_spearman:
        latex_rows = []
        for dist in distance_cols:
            row = {"Metric": get_metric_label(dist)}
            for fid in fidelity_cols:
                row[f"$\\rho$ ({fid.upper()})"] = corr_spearman[fid][dist]["rho"]
            latex_rows.append(row)

        latex_df = pd.DataFrame(latex_rows).set_index("Metric")
        latex_str = to_latex_table(
            latex_df,
            caption="Spearman correlation between distance metrics and fidelity (all data)",
            label="correlation_overall",
            fmt=".3f",
        )
        latex_path = os.path.join(FIGURES_DIR, "correlation_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex_str)
        print(f"\nLaTeX table saved to {latex_path}")


if __name__ == "__main__":
    main()

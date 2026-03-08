"""Generate comprehensive CCIS analysis report with qualitative examples.

Reads combined_perframe.csv and produces:
  1. Full statistical report (text + LaTeX)
  2. Qualitative scatter plots (distance vs fidelity per metric)
  3. Representative examples: best/worst predicted frames
  4. Per-scene breakdowns with visual examples
  5. Metric agreement analysis
  6. Summary dashboard figure

Usage:
    python generate_report.py --csv ../results/perframe/combined_perframe.csv
    python generate_report.py --csv ../results/perframe/combined_perframe.csv --image_dir /gpfs/workdir/malhotraa/data
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import (
    ALL_METRICS, GEOMETRIC_METRICS, VISUAL_METRICS, NEW_METRICS, ORIGINAL_METRICS,
    METRIC_LABELS, DATASET_LABELS, DATASET_COLORS,
    get_metric_label, get_metric_color, savefig, to_latex_table, FIGURES_DIR,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =========================================================================
# 1. Scatter plots: distance vs fidelity
# =========================================================================

def plot_scatter_grid(df, distance_cols, fidelity_col="psnr"):
    """Grid of scatter plots: each distance metric vs fidelity, colored by dataset."""
    n_metrics = len(distance_cols)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

    datasets = df["dataset"].unique()

    for idx, metric in enumerate(distance_cols):
        ax = axes[idx]
        valid = df[[metric, fidelity_col, "dataset"]].dropna()

        for ds in sorted(datasets):
            ds_data = valid[valid["dataset"] == ds]
            ax.scatter(
                ds_data[metric], ds_data[fidelity_col],
                c=DATASET_COLORS.get(ds, "#999"),
                s=8, alpha=0.4,
                label=DATASET_LABELS.get(ds, ds),
            )

        # Correlation annotation
        if len(valid) > 10:
            rho, p = stats.spearmanr(valid[metric], valid[fidelity_col])
            ax.annotate(
                f"$\\rho$={rho:.3f}",
                xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=9, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

        ax.set_xlabel(get_metric_label(metric), fontsize=8)
        ax.set_ylabel(fidelity_col.upper(), fontsize=8)
        ax.tick_params(labelsize=7)

    # Legend on first subplot
    if len(axes) > 0:
        axes[0].legend(fontsize=7, markerscale=2)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Distance Metrics vs {fidelity_col.upper()}", fontsize=14, y=1.01)
    plt.tight_layout()
    savefig(fig, f"scatter_grid_{fidelity_col}")


# =========================================================================
# 2. Representative qualitative examples
# =========================================================================

def find_representative_examples(df, distance_cols, fidelity_col="psnr", n_examples=5):
    """Find frames that are well/poorly predicted by distance metrics."""
    # Train a quick XGBoost to get residuals
    available = [c for c in distance_cols if c in df.columns and df[c].notna().sum() > 100]
    subset = df[available + [fidelity_col, "dataset", "scene", "experiment", "frame_id"]].dropna()

    if len(subset) < 50:
        print("  Not enough data for qualitative examples")
        return None

    X = subset[available].values
    y = subset[fidelity_col].values

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    y_pred = model.predict(X)
    residuals = y - y_pred

    subset = subset.copy()
    subset["predicted"] = y_pred
    subset["residual"] = residuals
    subset["abs_residual"] = np.abs(residuals)

    # Categories of interest
    examples = {}

    # Well-predicted high quality (high PSNR, low residual)
    high_q = subset[subset[fidelity_col] > subset[fidelity_col].quantile(0.75)]
    examples["high_quality_well_predicted"] = high_q.nsmallest(n_examples, "abs_residual")

    # Well-predicted low quality
    low_q = subset[subset[fidelity_col] < subset[fidelity_col].quantile(0.25)]
    examples["low_quality_well_predicted"] = low_q.nsmallest(n_examples, "abs_residual")

    # Poorly predicted (largest |residual|)
    examples["worst_predicted"] = subset.nlargest(n_examples, "abs_residual")

    # Overestimated (model predicts high, actual is low)
    examples["overestimated"] = subset.nsmallest(n_examples, "residual")

    # Underestimated (model predicts low, actual is high)
    examples["underestimated"] = subset.nlargest(n_examples, "residual")

    return examples


def plot_representative_examples(examples, fidelity_col="psnr"):
    """Plot representative examples as annotated tables."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for idx, (category, ex_df) in enumerate(examples.items()):
        ax = axes[idx]
        ax.axis("off")
        title = category.replace("_", " ").title()
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

        # Create table
        table_data = []
        for _, row in ex_df.iterrows():
            table_data.append([
                row.get("dataset", ""),
                row.get("scene", ""),
                int(row.get("frame_id", 0)),
                f"{row.get(fidelity_col, 0):.2f}",
                f"{row.get('predicted', 0):.2f}",
                f"{row.get('residual', 0):+.2f}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["Dataset", "Scene", "Frame", f"Actual {fidelity_col.upper()}", "Predicted", "Residual"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        # Color residuals
        for i, row_data in enumerate(table_data):
            resid = float(row_data[-1])
            if abs(resid) < 1.0:
                color = "#c6efce"  # green
            elif abs(resid) < 3.0:
                color = "#ffeb9c"  # yellow
            else:
                color = "#ffc7ce"  # red
            table[i + 1, 5].set_facecolor(color)

    plt.tight_layout()
    savefig(fig, f"representative_examples_{fidelity_col}")


# =========================================================================
# 3. Metric agreement analysis
# =========================================================================

def plot_metric_agreement(df, distance_cols):
    """Analyze how well different distance metrics agree with each other."""
    available = [c for c in distance_cols if c in df.columns and df[c].notna().sum() > 100]
    valid = df[available].dropna()

    if len(valid) < 50:
        print("  Not enough data for metric agreement")
        return

    # Rank correlation between metrics
    n = len(available)
    rank_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = stats.spearmanr(valid[available[i]], valid[available[j]])
            rank_corr[i, j] = rho

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(rank_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = [get_metric_label(m) for m in available]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(rank_corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{rank_corr[i,j]:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Inter-Metric Rank Correlation (Spearman)")
    plt.tight_layout()
    savefig(fig, "metric_agreement")


# =========================================================================
# 4. Per-scene breakdown
# =========================================================================

def plot_perscene_breakdown(df, distance_cols, fidelity_col="psnr"):
    """Per-scene average fidelity + dominant distance metric."""
    available = [c for c in distance_cols if c in df.columns and df[c].notna().sum() > 100]

    scene_stats = []
    for (ds, scene), group in df.groupby(["dataset", "scene"]):
        row = {
            "dataset": ds,
            "scene": scene,
            "n_frames": len(group),
            "mean_psnr": group[fidelity_col].mean() if fidelity_col in group else np.nan,
            "std_psnr": group[fidelity_col].std() if fidelity_col in group else np.nan,
        }

        # Find best-correlated metric for this scene
        best_rho = 0
        best_metric = ""
        for metric in available:
            valid = group[[metric, fidelity_col]].dropna()
            if len(valid) >= 5:
                rho, _ = stats.spearmanr(valid[metric], valid[fidelity_col])
                if abs(rho) > abs(best_rho):
                    best_rho = rho
                    best_metric = metric

        row["best_metric"] = get_metric_label(best_metric) if best_metric else "N/A"
        row["best_rho"] = best_rho
        scene_stats.append(row)

    scene_df = pd.DataFrame(scene_stats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(scene_stats) * 0.25)))

    # Left: mean PSNR per scene
    scene_df_sorted = scene_df.sort_values("mean_psnr", ascending=True)
    colors = [DATASET_COLORS.get(ds, "#999") for ds in scene_df_sorted["dataset"]]
    ax1.barh(range(len(scene_df_sorted)), scene_df_sorted["mean_psnr"],
             xerr=scene_df_sorted["std_psnr"], color=colors, capsize=3)
    ax1.set_yticks(range(len(scene_df_sorted)))
    ax1.set_yticklabels([f"{r.scene} ({r.dataset})" for _, r in scene_df_sorted.iterrows()], fontsize=8)
    ax1.set_xlabel(f"Mean {fidelity_col.upper()}")
    ax1.set_title("Per-Scene Fidelity")

    # Legend
    handles = [mpatches.Patch(color=c, label=DATASET_LABELS.get(d, d))
               for d, c in DATASET_COLORS.items() if d in scene_df["dataset"].values]
    ax1.legend(handles=handles, fontsize=8)

    # Right: best-correlated metric per scene
    ax2.barh(range(len(scene_df_sorted)), scene_df_sorted["best_rho"].abs(), color=colors)
    ax2.set_yticks(range(len(scene_df_sorted)))
    ax2.set_yticklabels([f"{r.best_metric}" for _, r in scene_df_sorted.iterrows()], fontsize=7)
    ax2.set_xlabel("|Spearman ρ|")
    ax2.set_title("Best-Correlated Metric per Scene")

    plt.tight_layout()
    savefig(fig, f"perscene_breakdown_{fidelity_col}")

    return scene_df


# =========================================================================
# 5. Summary dashboard
# =========================================================================

def plot_dashboard(df, distance_cols, fidelity_col="psnr"):
    """Single-figure dashboard summarizing key findings."""
    available = [c for c in distance_cols if c in df.columns and df[c].notna().sum() > 100]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Overall correlation bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    rhos = []
    for m in available:
        valid = df[[m, fidelity_col]].dropna()
        if len(valid) > 10:
            rho, _ = stats.spearmanr(valid[m], valid[fidelity_col])
            rhos.append((m, rho))
    rhos.sort(key=lambda x: abs(x[1]), reverse=True)

    if rhos:
        names, vals = zip(*rhos)
        colors = [get_metric_color(n) for n in names]
        labels = [get_metric_label(n) for n in names]
        ax_a.barh(range(len(vals)), [abs(v) for v in vals], color=colors)
        ax_a.set_yticks(range(len(vals)))
        ax_a.set_yticklabels(labels, fontsize=7)
        ax_a.set_xlabel("|Spearman ρ|")
        ax_a.set_title("A. Correlation Ranking")

    # Panel B: Geometric vs Visual boxplot
    ax_b = fig.add_subplot(gs[0, 1])
    geo_rhos = []
    vis_rhos = []
    for m, rho in rhos:
        if m in GEOMETRIC_METRICS:
            geo_rhos.append(abs(rho))
        elif m in VISUAL_METRICS:
            vis_rhos.append(abs(rho))
    if geo_rhos or vis_rhos:
        bp = ax_b.boxplot(
            [geo_rhos if geo_rhos else [0], vis_rhos if vis_rhos else [0]],
            labels=["Geometric", "Visual"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("#1f77b4")
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor("#ff7f0e")
        ax_b.set_ylabel("|Spearman ρ|")
        ax_b.set_title("B. Geometric vs Visual")

    # Panel C: Dataset comparison
    ax_c = fig.add_subplot(gs[0, 2])
    datasets = sorted(df["dataset"].unique())
    if rhos:
        top_metric = rhos[0][0]  # Best overall metric
        for ds in datasets:
            ds_data = df[df["dataset"] == ds]
            valid = ds_data[[top_metric, fidelity_col]].dropna()
            if len(valid) > 10:
                ax_c.scatter(
                    valid[top_metric], valid[fidelity_col],
                    c=DATASET_COLORS.get(ds, "#999"),
                    s=10, alpha=0.4,
                    label=DATASET_LABELS.get(ds, ds),
                )
        ax_c.set_xlabel(get_metric_label(top_metric))
        ax_c.set_ylabel(fidelity_col.upper())
        ax_c.set_title(f"C. Best Metric by Dataset")
        ax_c.legend(fontsize=8)

    # Panel D: New vs Original metrics R² comparison
    ax_d = fig.add_subplot(gs[1, 0])
    orig_available = [m for m in ORIGINAL_METRICS if m in available]
    new_available = [m for m in NEW_METRICS if m in available]
    all_available = available

    r2_results = {}
    for label, cols in [("All", all_available), ("Original 7", orig_available), ("New 5", new_available)]:
        if not cols:
            continue
        subset = df[cols + [fidelity_col]].dropna()
        if len(subset) < 50:
            continue
        X = subset[cols].values
        y = subset[fidelity_col].values
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X, y, verbose=False)
        y_pred = model.predict(X)
        r2_results[label] = r2_score(y, y_pred)

    if r2_results:
        bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][:len(r2_results)]
        ax_d.bar(range(len(r2_results)), list(r2_results.values()), color=bar_colors)
        ax_d.set_xticks(range(len(r2_results)))
        ax_d.set_xticklabels(list(r2_results.keys()))
        ax_d.set_ylabel("R² (train set)")
        ax_d.set_title("D. Feature Set Comparison")

    # Panel E: Distribution of PSNR by method
    ax_e = fig.add_subplot(gs[1, 1])
    if "method" in df.columns:
        methods = sorted(df["method"].unique())[:6]
        data = [df[df["method"] == m][fidelity_col].dropna().values for m in methods]
        data = [d for d in data if len(d) > 0]
        methods = [m for m, d in zip(methods, [df[df["method"] == m][fidelity_col].dropna() for m in methods]) if len(d) > 0]
        if data:
            ax_e.boxplot(data, labels=methods, patch_artist=True)
            ax_e.set_ylabel(fidelity_col.upper())
            ax_e.set_title("E. Fidelity by Method")
            ax_e.tick_params(axis="x", rotation=30, labelsize=8)

    # Panel F: Prediction scatter (actual vs predicted)
    ax_f = fig.add_subplot(gs[1, 2])
    if all_available:
        subset = df[all_available + [fidelity_col, "dataset"]].dropna()
        if len(subset) > 50:
            X = subset[all_available].values
            y = subset[fidelity_col].values
            model = xgb.XGBRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X, y, verbose=False)
            y_pred = model.predict(X)

            for ds in sorted(subset["dataset"].unique()):
                mask = (subset["dataset"] == ds).values
                ax_f.scatter(y[mask], y_pred[mask],
                           c=DATASET_COLORS.get(ds, "#999"),
                           s=8, alpha=0.3,
                           label=DATASET_LABELS.get(ds, ds))

            lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
            ax_f.plot(lims, lims, "k--", lw=1)
            ax_f.set_xlabel(f"Actual {fidelity_col.upper()}")
            ax_f.set_ylabel(f"Predicted {fidelity_col.upper()}")
            ax_f.set_title(f"F. Actual vs Predicted (R²={r2_score(y, y_pred):.3f})")
            ax_f.legend(fontsize=7)

    plt.suptitle("CCIS Extended Analysis Dashboard", fontsize=14, fontweight="bold")
    savefig(fig, f"dashboard_{fidelity_col}")


# =========================================================================
# 6. Text report
# =========================================================================

def generate_text_report(df, distance_cols, fidelity_col="psnr"):
    """Generate comprehensive text report."""
    available = [c for c in distance_cols if c in df.columns and df[c].notna().sum() > 10]

    lines = []
    lines.append("=" * 70)
    lines.append("  EXTENDED CCIS ANALYSIS REPORT")
    lines.append("  From Distance to Fidelity: Extended Analysis")
    lines.append("=" * 70)

    # Overview
    lines.append(f"\n1. DATA OVERVIEW")
    lines.append(f"   Total rows: {len(df)}")
    lines.append(f"   Datasets: {sorted(df['dataset'].unique())}")
    lines.append(f"   Scenes: {df['scene'].nunique()}")
    if "method" in df.columns:
        lines.append(f"   Methods: {sorted(df['method'].unique())}")
    if "budget" in df.columns:
        lines.append(f"   Budgets: {sorted(df['budget'].unique())}")
    lines.append(f"   Available distance metrics: {len(available)}")
    lines.append(f"   Fidelity columns: {[c for c in ['psnr', 'ssim', 'lpips'] if c in df.columns]}")

    # Completeness
    n_fid = df[fidelity_col].notna().sum()
    lines.append(f"\n   Completeness:")
    lines.append(f"   - {fidelity_col}: {n_fid} ({100*n_fid/len(df):.1f}%)")
    for m in available:
        n = df[m].notna().sum()
        lines.append(f"   - {get_metric_label(m)}: {n} ({100*n/len(df):.1f}%)")

    # Correlations
    lines.append(f"\n2. SPEARMAN CORRELATIONS (vs {fidelity_col.upper()})")
    corr_results = []
    for m in available:
        valid = df[[m, fidelity_col]].dropna()
        if len(valid) > 10:
            rho, p = stats.spearmanr(valid[m], valid[fidelity_col])
            corr_results.append((m, rho, p, len(valid)))

    corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
    for m, rho, p, n in corr_results:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"   {get_metric_label(m):25s}: ρ={rho:+.4f}  p={p:.2e}  n={n:5d}  {sig}")

    # Geometric vs Visual
    geo_rhos = [abs(rho) for m, rho, _, _ in corr_results if m in GEOMETRIC_METRICS]
    vis_rhos = [abs(rho) for m, rho, _, _ in corr_results if m in VISUAL_METRICS]
    if geo_rhos and vis_rhos:
        lines.append(f"\n   Geometric metrics avg |ρ|: {np.mean(geo_rhos):.4f}")
        lines.append(f"   Visual metrics avg |ρ|:    {np.mean(vis_rhos):.4f}")
        dominant = "GEOMETRIC" if np.mean(geo_rhos) > np.mean(vis_rhos) else "VISUAL"
        lines.append(f"   -> {dominant} metrics are more predictive")

    # Per-dataset
    lines.append(f"\n3. PER-DATASET CORRELATIONS")
    for ds in sorted(df["dataset"].unique()):
        ds_data = df[df["dataset"] == ds]
        lines.append(f"\n   {DATASET_LABELS.get(ds, ds)} ({len(ds_data)} frames):")
        for m, _, _, _ in corr_results[:5]:
            valid = ds_data[[m, fidelity_col]].dropna()
            if len(valid) > 10:
                rho, _ = stats.spearmanr(valid[m], valid[fidelity_col])
                lines.append(f"     {get_metric_label(m):25s}: ρ={rho:+.4f}  n={len(valid)}")

    # New vs Original
    lines.append(f"\n4. NEW vs ORIGINAL METRICS")
    new_rhos = [(m, rho) for m, rho, _, _ in corr_results if m in NEW_METRICS]
    orig_rhos = [(m, rho) for m, rho, _, _ in corr_results if m in ORIGINAL_METRICS]
    lines.append(f"   Original metrics ({len(orig_rhos)}):")
    for m, rho in orig_rhos:
        lines.append(f"     {get_metric_label(m):25s}: ρ={rho:+.4f}")
    lines.append(f"   New metrics ({len(new_rhos)}):")
    for m, rho in new_rhos:
        lines.append(f"     {get_metric_label(m):25s}: ρ={rho:+.4f}")

    # Key findings
    lines.append(f"\n5. KEY FINDINGS")
    if corr_results:
        best = corr_results[0]
        lines.append(f"   - Best predictor: {get_metric_label(best[0])} (ρ={best[1]:+.4f})")

        plucker_result = next((r for r in corr_results if r[0] == "fvs_plucker"), None)
        if plucker_result:
            rank = [r[0] for r in corr_results].index("fvs_plucker") + 1
            lines.append(f"   - Plucker distance ranks #{rank} (ρ={plucker_result[1]:+.4f})")

        infomax_result = next((r for r in corr_results if r[0] == "infomax3d_marginal"), None)
        if infomax_result:
            rank = [r[0] for r in corr_results].index("infomax3d_marginal") + 1
            lines.append(f"   - InfoMax3D marginal ranks #{rank} (ρ={infomax_result[1]:+.4f})")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive CCIS analysis report")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default="psnr", choices=["psnr", "ssim", "lpips"])
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    distance_cols = [c for c in ALL_METRICS if c in df.columns and df[c].notna().sum() > 10]

    section("DATA LOADED")
    print(f"  Rows: {len(df)}")
    print(f"  Distance metrics: {len(distance_cols)}")
    print(f"  Columns: {list(df.columns)}")

    section("1. SCATTER PLOTS")
    plot_scatter_grid(df, distance_cols, args.target)

    section("2. REPRESENTATIVE EXAMPLES")
    examples = find_representative_examples(df, distance_cols, args.target)
    if examples:
        plot_representative_examples(examples, args.target)
        # Print examples
        for category, ex_df in examples.items():
            print(f"\n  {category.upper()}:")
            for _, row in ex_df.iterrows():
                print(f"    {row.get('dataset','')}/{row.get('scene','')}/frame_{int(row.get('frame_id',0))}: "
                      f"{args.target}={row.get(args.target,0):.2f} pred={row.get('predicted',0):.2f} "
                      f"resid={row.get('residual',0):+.2f}")

    section("3. METRIC AGREEMENT")
    plot_metric_agreement(df, distance_cols)

    section("4. PER-SCENE BREAKDOWN")
    scene_df = plot_perscene_breakdown(df, distance_cols, args.target)
    if scene_df is not None:
        print(scene_df.to_string(index=False))

    section("5. SUMMARY DASHBOARD")
    plot_dashboard(df, distance_cols, args.target)

    section("6. TEXT REPORT")
    report = generate_text_report(df, distance_cols, args.target)
    print(report)

    # Save report
    report_path = os.path.join(FIGURES_DIR, "ccis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to {report_path}")

    # Summary of all figures
    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    for f_name in sorted(os.listdir(FIGURES_DIR)):
        if f_name.endswith((".pdf", ".png")):
            path = os.path.join(FIGURES_DIR, f_name)
            size_kb = os.path.getsize(path) / 1024
            print(f"    {f_name:45s} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

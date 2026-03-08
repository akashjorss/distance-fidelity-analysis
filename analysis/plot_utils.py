"""Shared plotting utilities, styling, and LaTeX export helpers for CCIS analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# Output directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# LaTeX-friendly styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color scheme
METRIC_COLORS = {
    "fvs_baseline": "#1f77b4",
    "fvs_plucker": "#ff7f0e",
    "fvs_angular": "#2ca02c",
    "fvs_euclidean": "#d62728",
    "pc_max": "#9467bd",
    "conmax3d_cov": "#8c564b",
    "alexnet_entropy": "#e377c2",
    "alexnet_dist": "#7f7f7f",
    "dinov2_dist": "#bcbd22",
    "clip_dist": "#17becf",
    "infomax3d_marginal": "#ff9896",
    "lpips_dist": "#aec7e8",
}

METRIC_LABELS = {
    "fvs_baseline": "FVS (baseline)",
    "fvs_plucker": r"FVS-Pl\"ucker",
    "fvs_angular": "FVS-Angular",
    "fvs_euclidean": "FVS-Euclidean",
    "pc_max": "PC-Max",
    "conmax3d_cov": "ConMax3D Cov.",
    "alexnet_entropy": "AlexNet Entropy",
    "alexnet_dist": "AlexNet Dist.",
    "dinov2_dist": "DINOv2 Dist.",
    "clip_dist": "CLIP Dist.",
    "infomax3d_marginal": "InfoMax3D Marg.",
    "lpips_dist": "LPIPS Dist.",
}

DATASET_LABELS = {
    "llff": "LLFF",
    "tt": "T\\&T",
    "ns": "NeRF Synthetic",
}

DATASET_COLORS = {
    "llff": "#1f77b4",
    "tt": "#ff7f0e",
    "ns": "#2ca02c",
}

GEOMETRIC_METRICS = ["fvs_baseline", "fvs_plucker", "fvs_angular", "fvs_euclidean"]
VISUAL_METRICS = ["alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist"]
NEW_METRICS = ["fvs_plucker", "fvs_angular", "fvs_euclidean", "infomax3d_marginal", "lpips_dist"]
ORIGINAL_METRICS = ["fvs_baseline", "pc_max", "conmax3d_cov", "alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist"]
ALL_METRICS = list(METRIC_LABELS.keys())


def get_metric_label(metric):
    return METRIC_LABELS.get(metric, metric)


def get_metric_color(metric):
    return METRIC_COLORS.get(metric, "#333333")


def savefig(fig, name, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f"{name}.{fmt}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def correlation_heatmap(corr_matrix, title="", output_name="correlation"):
    """Plot a correlation heatmap."""
    metrics = corr_matrix.columns.tolist()
    n = len(metrics)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels([get_metric_label(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([get_metric_label(m) for m in metrics])

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)

    savefig(fig, output_name)
    return fig


def to_latex_table(df, caption="", label="", bold_best=True, fmt=".3f"):
    """Convert DataFrame to LaTeX table string."""
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:" + label + "}")

    cols = "l" + "r" * (len(df.columns))
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    # Header
    header = " & ".join([""] + [str(c) for c in df.columns]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best values per column
    if bold_best:
        best_idx = {}
        for col in df.columns:
            try:
                vals = df[col].astype(float)
                best_idx[col] = vals.abs().idxmax()
            except (ValueError, TypeError):
                pass

    # Rows
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for col in df.columns:
            val = row[col]
            try:
                val_str = f"{float(val):{fmt}}"
            except (ValueError, TypeError):
                val_str = str(val)

            if bold_best and col in best_idx and idx == best_idx[col]:
                val_str = r"\textbf{" + val_str + "}"
            cells.append(val_str)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

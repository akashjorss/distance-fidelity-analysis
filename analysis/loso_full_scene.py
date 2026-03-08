"""LOSO: Leave out entire scene (all budgets) and test generalization.

Compares z-scored vs raw PSNR prediction.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import METRIC_COLORS, DATASET_COLORS, savefig, get_metric_label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISTANCE_COLS = [
    "fvs_baseline", "fvs_plucker", "fvs_angular", "fvs_euclidean",
    "alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist",
]


def run_loso_full_scene(df, target="psnr", normalize=True):
    """Leave out entire scene (all k values). Optionally z-score normalize."""
    available = [c for c in DISTANCE_COLS if df[c].notna().sum() > 100]
    features = available + ["budget"]
    df_clean = df.dropna(subset=[target] + available).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)

    if normalize:
        stats_df = df_clean.groupby("scene")[target].agg(["mean", "std"]).reset_index()
        stats_df.columns = ["scene", "s_mean", "s_std"]
        df_clean = df_clean.merge(stats_df, on="scene")
        df_clean["target_val"] = (df_clean[target] - df_clean["s_mean"]) / df_clean["s_std"].clip(lower=1e-6)
    else:
        df_clean["target_val"] = df_clean[target]

    results = []
    for scene in sorted(df_clean["scene"].unique()):
        test = df_clean[df_clean["scene"] == scene]
        train = df_clean[df_clean["scene"] != scene]
        if len(test) < 10:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        model.fit(train[features].fillna(0), train["target_val"].values)
        pred = model.predict(test[features].fillna(0))
        y = test["target_val"].values

        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rho, _ = stats.spearmanr(pred, y)
        mae = np.mean(np.abs(y - pred))

        # Also get ρ on original scale
        rho_orig, _ = stats.spearmanr(pred, test[target].values)

        results.append({
            "scene": scene, "dataset": test["dataset"].iloc[0],
            "n": len(test), "r2": r2, "rho": rho, "rho_orig": rho_orig, "mae": mae,
        })

    return pd.DataFrame(results), features


def plot_comparison(raw_df, zscore_df, target, outdir):
    """Side-by-side comparison of raw vs z-scored LOSO."""
    plt.rcParams.update({"font.size": 10})

    scenes = sorted(set(raw_df["scene"]) | set(zscore_df["scene"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # R² comparison
    ax = axes[0]
    raw_map = dict(zip(raw_df["scene"], raw_df["r2"]))
    z_map = dict(zip(zscore_df["scene"], zscore_df["r2"]))
    y_pos = range(len(scenes))
    ax.barh([y - 0.2 for y in y_pos], [raw_map.get(s, 0) for s in scenes],
            height=0.35, color="salmon", label="Raw PSNR")
    ax.barh([y + 0.2 for y in y_pos], [z_map.get(s, 0) for s in scenes],
            height=0.35, color="green", label="Z-scored PSNR")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(scenes)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("R²")
    ax.set_title("LOSO R²: Raw vs Z-scored")
    ax.legend(loc="lower right")

    # ρ comparison
    ax = axes[1]
    raw_rho = dict(zip(raw_df["scene"], raw_df["rho"]))
    z_rho = dict(zip(zscore_df["scene"], zscore_df["rho"]))
    ax.barh([y - 0.2 for y in y_pos], [raw_rho.get(s, 0) for s in scenes],
            height=0.35, color="salmon", label="Raw PSNR")
    ax.barh([y + 0.2 for y in y_pos], [z_rho.get(s, 0) for s in scenes],
            height=0.35, color="green", label="Z-scored PSNR")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(scenes)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("LOSO ρ: Raw vs Z-scored")
    ax.legend(loc="lower right")

    # Scatter: predicted z-score vs actual z-score (pooled)
    ax = axes[2]
    # Re-run to collect predictions for scatter
    ax.text(0.5, 0.5, f"Z-scored LOSO\nMean R²={zscore_df['r2'].mean():.3f}\nMean ρ={zscore_df['rho'].mean():.3f}\n\nRaw LOSO\nMean R²={raw_df['r2'].mean():.3f}\nMean ρ={raw_df['rho'].mean():.3f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary")
    ax.axis("off")

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"loso_full_scene_comparison_{target}"))


def plot_feature_importance(df_clean, target, features, outdir):
    """Train on all data with z-score, show feature importance."""
    plt.rcParams.update({"font.size": 10})

    available = [c for c in DISTANCE_COLS if c in df_clean.columns and df_clean[c].notna().sum() > 100]
    feats = available + ["budget"]

    stats_df = df_clean.groupby("scene")[target].agg(["mean", "std"]).reset_index()
    stats_df.columns = ["scene", "s_mean", "s_std"]
    df_z = df_clean.merge(stats_df, on="scene")
    df_z["target_z"] = (df_z[target] - df_z["s_mean"]) / df_z["s_std"].clip(lower=1e-6)
    df_z = df_z.dropna(subset=available)
    df_z["budget"] = df_z["budget"].astype(float)

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(df_z[feats].fillna(0), df_z["target_z"].values)

    imp = dict(zip(feats, model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [get_metric_label(k) if k != "budget" else "budget" for k, _ in imp_sorted]
    vals = [v for _, v in imp_sorted]
    colors = [METRIC_COLORS.get(k, "#999") if k != "budget" else "#666" for k, _ in imp_sorted]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost Feature Importance for z-scored {target.upper()} (no scene_id)")
    ax.invert_yaxis()
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"loso_full_scene_importance_{target}"))

    return imp_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="psnr")
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    target = args.target
    print(f"Loaded {len(df)} rows, target={target}")

    # Raw PSNR LOSO
    print("\n=== LOSO Full Scene: RAW PSNR ===")
    raw_df, features = run_loso_full_scene(df, target, normalize=False)
    for _, r in raw_df.iterrows():
        print(f"  {r['scene']:12s} ({r['dataset']:4s}, n={r['n']:5d}): R²={r['r2']:+.4f}  ρ={r['rho']:.4f}")
    print(f"\n  Mean R²: {raw_df['r2'].mean():.4f}")
    print(f"  Mean ρ:  {raw_df['rho'].mean():.4f}")
    print(f"  R²>0: {(raw_df['r2']>0).sum()}/{len(raw_df)}")

    # Z-scored PSNR LOSO
    print("\n=== LOSO Full Scene: Z-SCORED PSNR ===")
    zscore_df, _ = run_loso_full_scene(df, target, normalize=True)
    for _, r in zscore_df.iterrows():
        print(f"  {r['scene']:12s} ({r['dataset']:4s}, n={r['n']:5d}): R²(z)={r['r2']:+.4f}  ρ(z)={r['rho']:.4f}")
    print(f"\n  Mean R²(z): {zscore_df['r2'].mean():.4f}")
    print(f"  Mean ρ(z):  {zscore_df['rho'].mean():.4f}")
    print(f"  R²(z)>0: {(zscore_df['r2']>0).sum()}/{len(zscore_df)}")

    # Per-dataset breakdown
    print("\n  Per-dataset:")
    for d in sorted(zscore_df["dataset"].unique()):
        sub = zscore_df[zscore_df["dataset"] == d]
        print(f"    {d}: R²(z)={sub['r2'].mean():.4f}, ρ(z)={sub['rho'].mean():.4f}, "
              f"R²>0: {(sub['r2']>0).sum()}/{len(sub)}")

    # Plots
    plot_comparison(raw_df, zscore_df, target, outdir)

    # Feature importance (z-scored, no scene_id)
    print("\n=== Feature Importance (z-scored model, no scene_id) ===")
    df_clean = df.dropna(subset=[target])
    imp = plot_feature_importance(df_clean, target, features, outdir)
    for name, val in imp:
        lbl = get_metric_label(name) if name != "budget" else "budget"
        print(f"  {lbl:20s}: {val:.4f}")

    # Save CSVs
    raw_df.to_csv(os.path.join(outdir, f"loso_full_scene_raw_{target}.csv"), index=False)
    zscore_df.to_csv(os.path.join(outdir, f"loso_full_scene_zscore_{target}.csv"), index=False)

    print(f"\nAll saved to {outdir}")


if __name__ == "__main__":
    main()

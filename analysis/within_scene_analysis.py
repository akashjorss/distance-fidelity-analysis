"""Within-scene and within-(scene,k) fidelity prediction analysis.

Tests whether distance metrics predict per-frame fidelity WITHIN a scene,
removing the scene-level PSNR baseline confound that dominates cross-scene models.

Analyses:
1. Per-scene Spearman correlations (pooled across methods/budgets within each scene)
2. Per-(scene,k) Spearman correlations (fixed scene AND budget, varying only method)
3. Within-scene XGBoost with 5-fold CV (no scene_id feature needed)
4. LOSO XGBoost on z-score normalized PSNR (removes per-scene scale)
5. Leave-one-(scene,k)-out XGBoost on z-scored PSNR

Usage:
    python analysis/within_scene_analysis.py --csv results/perframe/combined_perframe.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import (
    ALL_METRICS, METRIC_LABELS, METRIC_COLORS,
    DATASET_COLORS, savefig, get_metric_label,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISTANCE_COLS = [
    "fvs_baseline", "fvs_plucker", "fvs_angular", "fvs_euclidean",
    "alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist",
]


def label(col):
    return get_metric_label(col)


def per_scene_correlations(df, target="psnr"):
    """Spearman correlation of each metric vs target, computed per scene."""
    results = []
    for scene in sorted(df["scene"].unique()):
        sdf = df[df["scene"] == scene].dropna(subset=[target])
        if len(sdf) < 10:
            continue
        row = {"scene": scene, "dataset": sdf["dataset"].iloc[0], "n": len(sdf)}
        for col in DISTANCE_COLS:
            valid = sdf.dropna(subset=[col])
            if len(valid) < 10:
                row[col] = np.nan
            else:
                row[col], _ = stats.spearmanr(valid[col], valid[target])
        results.append(row)
    return pd.DataFrame(results)


def per_scene_k_correlations(df, target="psnr"):
    """Spearman correlation per (scene, budget) slice."""
    results = []
    for (scene, budget), sdf in df.groupby(["scene", "budget"]):
        sdf = sdf.dropna(subset=[target])
        if len(sdf) < 10:
            continue
        row = {
            "scene": scene, "budget": int(budget),
            "dataset": sdf["dataset"].iloc[0],
            "n": len(sdf), "n_methods": sdf["method"].nunique(),
        }
        for col in DISTANCE_COLS:
            valid = sdf.dropna(subset=[col])
            if len(valid) < 10:
                row[col] = np.nan
            else:
                row[col], _ = stats.spearmanr(valid[col], valid[target])
        results.append(row)
    return pd.DataFrame(results)


def within_scene_xgboost(df, target="psnr"):
    """XGBoost trained per scene (no scene_id), 5-fold CV."""
    from sklearn.model_selection import KFold
    import xgboost as xgb

    results = []
    for scene in sorted(df["scene"].unique()):
        sdf = df[df["scene"] == scene].dropna(subset=[target])
        available = [c for c in DISTANCE_COLS if sdf[c].notna().sum() > 10]
        if len(sdf) < 30 or len(available) < 3:
            continue

        X = sdf[available + ["budget"]].copy()
        X["budget"] = X["budget"].astype(float)
        X = X.fillna(0)
        y = sdf[target].values

        n_splits = min(5, max(2, len(sdf) // 20))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_r2, fold_rho = [], []

        for train_idx, test_idx in kf.split(X):
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbosity=0,
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])

            ss_res = np.sum((y[test_idx] - pred) ** 2)
            ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rho, _ = stats.spearmanr(pred, y[test_idx])
            fold_r2.append(r2)
            fold_rho.append(rho)

        results.append({
            "scene": scene, "dataset": sdf["dataset"].iloc[0], "n": len(sdf),
            "r2_mean": np.mean(fold_r2), "r2_std": np.std(fold_r2),
            "rho_mean": np.mean(fold_rho), "rho_std": np.std(fold_rho),
        })

    return pd.DataFrame(results)


def zscore_normalize(df, target="psnr"):
    """Z-score normalize target per scene: (x - scene_mean) / scene_std."""
    df = df.copy()
    ztarget = f"{target}_z"
    scene_stats = df.groupby("scene")[target].agg(["mean", "std"]).reset_index()
    scene_stats.columns = ["scene", "scene_mean", "scene_std"]
    df = df.merge(scene_stats, on="scene")
    df[ztarget] = (df[target] - df["scene_mean"]) / df["scene_std"].clip(lower=1e-6)
    return df, ztarget


def loso_zscore_xgboost(df, target="psnr"):
    """LOSO XGBoost on z-score normalized target. Removes per-scene scale."""
    import xgboost as xgb

    df_clean, ztarget = zscore_normalize(df.dropna(subset=[target]), target)
    available = [c for c in DISTANCE_COLS if df_clean[c].notna().sum() > 100]
    features = available + ["budget"]
    df_clean = df_clean.dropna(subset=available)
    df_clean["budget"] = df_clean["budget"].astype(float)

    results = []
    for scene in sorted(df_clean["scene"].unique()):
        test_df = df_clean[df_clean["scene"] == scene]
        train_df = df_clean[df_clean["scene"] != scene]
        if len(test_df) < 10 or len(train_df) < 50:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(train_df[features].fillna(0), train_df[ztarget].values)
        pred = model.predict(test_df[features].fillna(0))
        y_test = test_df[ztarget].values

        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rho, _ = stats.spearmanr(pred, y_test)
        mae = np.mean(np.abs(y_test - pred))

        # Also compute ρ on original scale (should match z-score ρ since it's monotonic)
        rho_orig, _ = stats.spearmanr(pred, test_df[target].values)

        results.append({
            "scene": scene, "dataset": test_df["dataset"].iloc[0],
            "n": len(test_df), "r2_z": r2, "rho_z": rho,
            "rho_orig": rho_orig, "mae_z": mae,
        })

    return pd.DataFrame(results)


def loso_scene_k_zscore(df, target="psnr"):
    """Leave-one-(scene,k)-out on z-scored target."""
    import xgboost as xgb

    df_clean, ztarget = zscore_normalize(df.dropna(subset=[target]), target)
    available = [c for c in DISTANCE_COLS if df_clean[c].notna().sum() > 100]
    features = available + ["budget"]
    df_clean = df_clean.dropna(subset=available)
    df_clean["budget"] = df_clean["budget"].astype(float)

    results = []
    for (scene, budget), test_df in df_clean.groupby(["scene", "budget"]):
        if len(test_df) < 10:
            continue
        train_df = df_clean[(df_clean["scene"] != scene) | (df_clean["budget"] != budget)]
        if len(train_df) < 50:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(train_df[features].fillna(0), train_df[ztarget].values)
        pred = model.predict(test_df[features].fillna(0))
        y_test = test_df[ztarget].values

        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rho, _ = stats.spearmanr(pred, y_test)

        results.append({
            "scene": scene, "budget": int(budget),
            "dataset": test_df["dataset"].iloc[0],
            "n": len(test_df), "r2": r2, "rho": rho,
        })

    return pd.DataFrame(results)


# ---- Plotting ----

def plot_per_scene_correlations(corr_df, target, outdir):
    plt.rcParams.update({"font.size": 10})
    cols = [c for c in DISTANCE_COLS if c in corr_df.columns]
    pivot = corr_df.set_index("scene")[cols]
    pivot.columns = [label(c) for c in cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label=f"Spearman ρ (vs {target.upper()})")
    ax.set_title(f"Per-Scene Correlations: Distance → {target.upper()}")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"within_scene_correlations_{target}"))


def plot_per_scene_k_summary(corr_sk_df, target, outdir):
    plt.rcParams.update({"font.size": 10})
    cols = [c for c in DISTANCE_COLS if c in corr_sk_df.columns]

    data = [corr_sk_df[c].dropna().values for c in cols]
    labels_list = [label(c) for c in cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels_list, vert=True, patch_artist=True)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(METRIC_COLORS.get(col, "#999999"))
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"Spearman ρ (vs {target.upper()})")
    ax.set_title(f"Per-(Scene, Budget) Correlations: Distance → {target.upper()}")
    ax.set_xticklabels(labels_list, rotation=45, ha="right")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"within_scene_k_correlations_{target}"))


def plot_within_scene_xgboost(xgb_df, target, outdir):
    plt.rcParams.update({"font.size": 10})
    xgb_df = xgb_df.sort_values("r2_mean", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (col, xlabel) in enumerate([("r2_mean", "R²"), ("rho_mean", "Spearman ρ")]):
        ax = axes[idx]
        colors = [DATASET_COLORS.get(d, "#999") for d in xgb_df["dataset"]]
        std_col = col.replace("mean", "std")
        ax.barh(range(len(xgb_df)), xgb_df[col], color=colors,
                xerr=xgb_df[std_col], capsize=3)
        ax.set_yticks(range(len(xgb_df)))
        ax.set_yticklabels(xgb_df["scene"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"{xlabel} (5-fold CV)")
        ax.set_title(f"Within-Scene XGBoost: {target.upper()}")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d) for d, c in DATASET_COLORS.items()]
    axes[1].legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"within_scene_xgboost_{target}"))


def plot_loso_zscore(loso_df, target, outdir):
    """Bar chart of LOSO R² on z-scored target."""
    plt.rcParams.update({"font.size": 10})
    loso_df = loso_df.sort_values("r2_z", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (col, xlabel) in enumerate([("r2_z", "R² (z-scored)"), ("rho_z", "ρ (z-scored)")]):
        ax = axes[idx]
        colors = ["green" if v > 0 else "red" for v in loso_df[col]]
        ax.barh(range(len(loso_df)), loso_df[col], color=colors)
        ax.set_yticks(range(len(loso_df)))
        ax.set_yticklabels(loso_df["scene"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"LOSO on z-scored {target.upper()}")

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"loso_zscore_{target}"))


def plot_loso_scene_k(loso_df, target, outdir):
    plt.rcParams.update({"font.size": 10})
    pivot_r2 = loso_df.pivot_table(index="scene", columns="budget", values="r2")
    pivot_rho = loso_df.pivot_table(index="scene", columns="budget", values="rho")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, pivot, metric_title, cmap, vrange in [
        (axes[0], pivot_r2, "R² (z-scored)", "RdYlGn", (-3, 1)),
        (axes[1], pivot_rho, "Spearman ρ", "RdBu_r", (-1, 1)),
    ]:
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                       aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"k={int(b)}" for b in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label=metric_title)
        ax.set_title(f"Leave-One-(Scene,k)-Out: {metric_title}")

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, f"loso_scene_k_{target}"))


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

    # 1. Per-scene correlations
    print("\n=== 1. Per-Scene Correlations ===")
    corr_df = per_scene_correlations(df, target)
    print(corr_df.to_string(index=False))
    plot_per_scene_correlations(corr_df, target, outdir)

    cols = [c for c in DISTANCE_COLS if c in corr_df.columns]
    print(f"\nMean |ρ| across scenes:")
    for c in cols:
        vals = corr_df[c].dropna()
        print(f"  {label(c):20s}: {vals.abs().mean():.4f} (median ρ: {vals.median():.4f})")

    # 2. Per-(scene, k) correlations
    print("\n=== 2. Per-(Scene, Budget) Correlations ===")
    corr_sk_df = per_scene_k_correlations(df, target)
    print(f"Total slices with n>=10: {len(corr_sk_df)}")
    print(f"\nMedian ρ per metric:")
    for c in cols:
        vals = corr_sk_df[c].dropna()
        print(f"  {label(c):20s}: median={vals.median():.4f}, mean={vals.mean():.4f}, n={len(vals)}")
    plot_per_scene_k_summary(corr_sk_df, target, outdir)

    # 3. Within-scene XGBoost
    print("\n=== 3. Within-Scene XGBoost (5-fold CV) ===")
    xgb_df = within_scene_xgboost(df, target)
    print(xgb_df.to_string(index=False))
    print(f"\nMean R²: {xgb_df['r2_mean'].mean():.4f}")
    print(f"Mean ρ:  {xgb_df['rho_mean'].mean():.4f}")
    print(f"Scenes with R²>0: {(xgb_df['r2_mean']>0).sum()}/{len(xgb_df)}")
    plot_within_scene_xgboost(xgb_df, target, outdir)

    # 4. LOSO on z-score normalized PSNR
    print("\n=== 4. LOSO XGBoost on z-scored PSNR ===")
    loso_z_df = loso_zscore_xgboost(df, target)
    print(loso_z_df.to_string(index=False))
    print(f"\nMean R²(z): {loso_z_df['r2_z'].mean():.4f}")
    print(f"Mean ρ(z):  {loso_z_df['rho_z'].mean():.4f}")
    print(f"Scenes with R²(z)>0: {(loso_z_df['r2_z']>0).sum()}/{len(loso_z_df)}")
    plot_loso_zscore(loso_z_df, target, outdir)

    # 5. Leave-one-(scene,k)-out on z-scored PSNR
    print("\n=== 5. Leave-One-(Scene,k)-Out on z-scored PSNR ===")
    loso_sk_df = loso_scene_k_zscore(df, target)
    print(f"Total slices: {len(loso_sk_df)}")
    print(f"R² > 0: {(loso_sk_df['r2']>0).sum()}/{len(loso_sk_df)}")
    print(f"Mean R²: {loso_sk_df['r2'].mean():.4f}")
    print(f"Mean ρ:  {loso_sk_df['rho'].mean():.4f}")
    for dataset in sorted(loso_sk_df["dataset"].unique()):
        ddf = loso_sk_df[loso_sk_df["dataset"] == dataset]
        print(f"  {dataset}: R²={ddf['r2'].mean():.4f}, ρ={ddf['rho'].mean():.4f}, "
              f"R²>0: {(ddf['r2']>0).sum()}/{len(ddf)}")
    plot_loso_scene_k(loso_sk_df, target, outdir)

    # Save CSV tables
    corr_df.to_csv(os.path.join(outdir, f"within_scene_corr_{target}.csv"), index=False)
    corr_sk_df.to_csv(os.path.join(outdir, f"within_scene_k_corr_{target}.csv"), index=False)
    xgb_df.to_csv(os.path.join(outdir, f"within_scene_xgb_{target}.csv"), index=False)
    loso_z_df.to_csv(os.path.join(outdir, f"loso_zscore_{target}.csv"), index=False)
    loso_sk_df.to_csv(os.path.join(outdir, f"loso_scene_k_{target}.csv"), index=False)

    print(f"\nAll figures and tables saved to {outdir}")


if __name__ == "__main__":
    main()

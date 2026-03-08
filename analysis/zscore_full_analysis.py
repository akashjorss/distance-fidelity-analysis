"""Full analysis pipeline on z-score normalized PSNR.

Removes per-scene baseline so all results reflect the distance-fidelity
relationship independent of scene difficulty.

Runs: correlations, XGBoost regression (5-fold CV), SHAP, feature importance,
binary classification, LOSO, cross-dataset generalization.
"""
import os, sys, argparse, json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import (
    METRIC_LABELS, METRIC_COLORS, DATASET_COLORS,
    GEOMETRIC_METRICS, VISUAL_METRICS,
    savefig, get_metric_label,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISTANCE_COLS = [
    "fvs_baseline", "fvs_plucker", "fvs_angular", "fvs_euclidean",
    "alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist",
]

# Drop fvs_euclidean since it's identical to fvs_baseline
DISTANCE_COLS_DEDUP = [
    "fvs_baseline", "fvs_plucker", "fvs_angular",
    "alexnet_entropy", "alexnet_dist", "dinov2_dist", "clip_dist",
]


def label(c):
    if c == "fvs_baseline":
        return "FVS-Euclidean"
    return get_metric_label(c)


def zscore_normalize(df, target="psnr"):
    df = df.copy()
    st = df.groupby("scene")[target].agg(["mean", "std"]).reset_index()
    st.columns = ["scene", "s_mean", "s_std"]
    df = df.merge(st, on="scene")
    df[f"{target}_z"] = (df[target] - df["s_mean"]) / df["s_std"].clip(lower=1e-6)
    return df


def run_correlations(df, ztarget, outdir):
    """Spearman/Pearson correlations on z-scored target."""
    print("\n" + "="*60)
    print("1. CORRELATIONS (z-scored)")
    print("="*60)

    cols = [c for c in DISTANCE_COLS_DEDUP if df[c].notna().sum() > 100]

    # Overall
    print("\nOverall Spearman correlations:")
    corr_results = []
    for c in cols:
        valid = df.dropna(subset=[c, ztarget])
        rho, p = stats.spearmanr(valid[c], valid[ztarget])
        r_pearson, _ = stats.pearsonr(valid[c], valid[ztarget])
        corr_results.append({"metric": c, "spearman": rho, "pearson": r_pearson, "n": len(valid)})
        print(f"  {label(c):20s}: ρ={rho:+.4f}  r={r_pearson:+.4f}  n={len(valid)}")

    # Per dataset
    print("\nPer-dataset Spearman:")
    for dataset in sorted(df["dataset"].unique()):
        ddf = df[df["dataset"] == dataset]
        print(f"\n  {dataset} ({len(ddf)} frames):")
        for c in cols:
            valid = ddf.dropna(subset=[c, ztarget])
            if len(valid) < 10:
                continue
            rho, _ = stats.spearmanr(valid[c], valid[ztarget])
            print(f"    {label(c):20s}: ρ={rho:+.4f}  n={len(valid)}")

    # Geometric vs Visual
    geo = [c for c in cols if c in GEOMETRIC_METRICS]
    vis = [c for c in cols if c in VISUAL_METRICS]
    geo_rhos = [abs(r["spearman"]) for r in corr_results if r["metric"] in geo]
    vis_rhos = [abs(r["spearman"]) for r in corr_results if r["metric"] in vis]
    print(f"\n  Geometric avg |ρ|: {np.mean(geo_rhos):.4f}")
    print(f"  Visual avg |ρ|:    {np.mean(vis_rhos):.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_results.sort(key=lambda x: abs(x["spearman"]))
    names = [label(r["metric"]) for r in corr_results]
    vals = [r["spearman"] for r in corr_results]
    colors = [METRIC_COLORS.get(r["metric"], "#999") for r in corr_results]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Spearman ρ (vs z-scored PSNR)")
    ax.set_title("Correlations with z-scored PSNR")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_correlations"))

    return corr_results


def run_xgboost(df, ztarget, outdir):
    """XGBoost 5-fold CV on z-scored target. Compare feature sets."""
    from sklearn.model_selection import KFold
    import xgboost as xgb

    print("\n" + "="*60)
    print("2. XGBOOST REGRESSION (z-scored, 5-fold CV)")
    print("="*60)

    cols = [c for c in DISTANCE_COLS_DEDUP if df[c].notna().sum() > 100]
    df_clean = df.dropna(subset=[ztarget] + cols).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)

    feature_sets = {
        "All 7 metrics + budget": cols + ["budget"],
        "Geometric only + budget": [c for c in cols if c in GEOMETRIC_METRICS] + ["budget"],
        "Visual only + budget": [c for c in cols if c in VISUAL_METRICS] + ["budget"],
        "All + scene_id + budget": cols + ["budget", "scene_id"],
    }

    # Encode scene_id
    scene_map = {s: i for i, s in enumerate(sorted(df_clean["scene"].unique()))}
    df_clean["scene_id"] = df_clean["scene"].map(scene_map)

    results_all = {}
    best_model = None
    best_r2 = -999

    for name, feats in feature_sets.items():
        available_feats = [f for f in feats if f in df_clean.columns]
        X = df_clean[available_feats].fillna(0)
        y = df_clean[ztarget].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_r2, fold_rho, fold_mae = [], [], []

        for train_idx, test_idx in kf.split(X):
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            y_test = y[test_idx]

            ss_res = np.sum((y_test - pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rho, _ = stats.spearmanr(pred, y_test)
            mae = np.mean(np.abs(y_test - pred))

            fold_r2.append(r2)
            fold_rho.append(rho)
            fold_mae.append(mae)

        r2_mean = np.mean(fold_r2)
        result = {
            "r2": f"{r2_mean:.4f} ± {np.std(fold_r2):.4f}",
            "rho": f"{np.mean(fold_rho):.4f} ± {np.std(fold_rho):.4f}",
            "mae": f"{np.mean(fold_mae):.4f} ± {np.std(fold_mae):.4f}",
            "r2_val": r2_mean,
            "rho_val": np.mean(fold_rho),
        }
        results_all[name] = result

        print(f"\n  {name}:")
        print(f"    R²:  {result['r2']}")
        print(f"    ρ:   {result['rho']}")
        print(f"    MAE: {result['mae']}")

        # Feature importance for this set
        model_full = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        model_full.fit(X, y)
        imp = dict(zip(available_feats, model_full.feature_importances_))
        imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        print("    Feature importance:")
        for feat, val in imp_sorted:
            print(f"      {label(feat) if feat not in ('budget','scene_id') else feat:20s}: {val:.4f}")

        if name == "All 7 metrics + budget" and r2_mean > best_r2:
            best_r2 = r2_mean
            best_model = model_full
            best_features = available_feats
            best_importance = imp_sorted

    # Plot feature importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of "All 7" feature importance
    ax = axes[0]
    names_imp = [label(k) if k not in ("budget", "scene_id") else k for k, _ in best_importance]
    vals_imp = [v for _, v in best_importance]
    colors_imp = [METRIC_COLORS.get(k, "#666") for k, _ in best_importance]
    ax.barh(range(len(names_imp)), vals_imp, color=colors_imp)
    ax.set_yticks(range(len(names_imp)))
    ax.set_yticklabels(names_imp)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("z-scored PSNR: All Metrics (no scene_id)")
    ax.invert_yaxis()

    # Feature set comparison bar chart
    ax = axes[1]
    set_names = list(results_all.keys())
    r2_vals = [results_all[n]["r2_val"] for n in set_names]
    rho_vals = [results_all[n]["rho_val"] for n in set_names]
    x = np.arange(len(set_names))
    ax.bar(x - 0.2, r2_vals, 0.35, label="R²", color="steelblue")
    ax.bar(x + 0.2, rho_vals, 0.35, label="ρ", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(set_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Feature Set Comparison (z-scored PSNR)")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_xgboost"))

    return results_all, best_model, best_features


def run_shap(model, df, ztarget, features, outdir):
    """SHAP analysis on the best model."""
    import shap

    print("\n" + "="*60)
    print("3. SHAP ANALYSIS (z-scored)")
    print("="*60)

    cols_needed = [c for c in DISTANCE_COLS_DEDUP if c in features]
    df_clean = df.dropna(subset=[ztarget] + cols_needed).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)
    X = df_clean[features].fillna(0)

    # Subsample for SHAP
    n_sample = min(5000, len(X))
    X_sample = X.sample(n_sample, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_imp = sorted(zip(features, mean_shap), key=lambda x: x[1], reverse=True)

    print("\n  Mean |SHAP| values:")
    total = sum(v for _, v in shap_imp)
    geo_total, vis_total = 0, 0
    for feat, val in shap_imp:
        lbl = label(feat) if feat not in ("budget", "scene_id") else feat
        pct = val / total * 100
        print(f"    {lbl:20s}: {val:.4f} ({pct:.1f}%)")
        if feat in GEOMETRIC_METRICS:
            geo_total += val
        elif feat in VISUAL_METRICS:
            vis_total += val

    print(f"\n  Geometric total: {geo_total:.4f} ({geo_total/total*100:.1f}%)")
    print(f"  Visual total:    {vis_total:.4f} ({vis_total/total*100:.1f}%)")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    # Rename features for display
    X_display = X_sample.copy()
    X_display.columns = [label(c) if c not in ("budget", "scene_id") else c for c in X_display.columns]
    shap.summary_plot(shap_values, X_display, show=False, max_display=len(features))
    plt.title("SHAP Values for z-scored PSNR Prediction")
    plt.tight_layout()
    savefig(plt.gcf(), os.path.join(outdir, "zscore_shap_beeswarm"))
    plt.close()

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names_s = [label(f) if f not in ("budget", "scene_id") else f for f, _ in shap_imp]
    vals_s = [v for _, v in shap_imp]
    colors_s = [METRIC_COLORS.get(f, "#666") for f, _ in shap_imp]
    ax.barh(range(len(names_s)), vals_s, color=colors_s)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance: z-scored PSNR")
    ax.invert_yaxis()
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_shap_bar"))

    return shap_imp


def run_binary_classification(df, ztarget, outdir):
    """Binary classification: z-scored PSNR > 0 (above scene average)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, roc_curve
    import xgboost as xgb

    print("\n" + "="*60)
    print("4. BINARY CLASSIFICATION (z-scored PSNR > 0 = above scene avg)")
    print("="*60)

    cols = [c for c in DISTANCE_COLS_DEDUP if df[c].notna().sum() > 100]
    df_clean = df.dropna(subset=[ztarget] + cols).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)
    features = cols + ["budget"]

    X = df_clean[features].fillna(0)
    y = (df_clean[ztarget] > 0).astype(int).values
    print(f"  Positive class (above avg): {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_auc = []
    all_y, all_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, eval_metric="logloss",
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], prob)
        fold_auc.append(auc)
        all_y.extend(y[test_idx])
        all_prob.extend(prob)

    print(f"  AUC-ROC: {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_y, all_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"AUC = {np.mean(fold_auc):.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary Classification: z-PSNR > 0 (above scene avg)")
    ax.legend()
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_binary_roc"))

    return np.mean(fold_auc)


def run_loso(df, ztarget, outdir):
    """LOSO on z-scored target."""
    import xgboost as xgb

    print("\n" + "="*60)
    print("5. LOSO (z-scored)")
    print("="*60)

    cols = [c for c in DISTANCE_COLS_DEDUP if df[c].notna().sum() > 100]
    features = cols + ["budget"]
    df_clean = df.dropna(subset=[ztarget] + cols).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)

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
        model.fit(train[features].fillna(0), train[ztarget].values)
        pred = model.predict(test[features].fillna(0))
        y = test[ztarget].values

        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rho, _ = stats.spearmanr(pred, y)

        results.append({
            "scene": scene, "dataset": test["dataset"].iloc[0],
            "n": len(test), "r2": r2, "rho": rho,
        })
        print(f"  {scene:12s} ({test['dataset'].iloc[0]:4s}): R²(z)={r2:+.4f}  ρ={rho:.4f}")

    rdf = pd.DataFrame(results)
    print(f"\n  Mean R²(z): {rdf['r2'].mean():.4f}")
    print(f"  Mean ρ:     {rdf['rho'].mean():.4f}")
    print(f"  R²>0: {(rdf['r2']>0).sum()}/{len(rdf)}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rdf_sorted = rdf.sort_values("r2")

    for idx, (col, xlabel) in enumerate([("r2", "R² (z-scored)"), ("rho", "Spearman ρ")]):
        ax = axes[idx]
        colors = ["green" if v > 0 else "red" for v in rdf_sorted[col]]
        ax.barh(range(len(rdf_sorted)), rdf_sorted[col], color=colors)
        ax.set_yticks(range(len(rdf_sorted)))
        ax.set_yticklabels(rdf_sorted["scene"])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"LOSO: {xlabel}")
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_loso"))

    return rdf


def run_cross_dataset(df, ztarget, outdir):
    """Train on one dataset, test on another — z-scored."""
    import xgboost as xgb

    print("\n" + "="*60)
    print("6. CROSS-DATASET GENERALIZATION (z-scored)")
    print("="*60)

    cols = [c for c in DISTANCE_COLS_DEDUP if df[c].notna().sum() > 100]
    features = cols + ["budget"]
    df_clean = df.dropna(subset=[ztarget] + cols).copy()
    df_clean["budget"] = df_clean["budget"].astype(float)

    datasets = sorted(df_clean["dataset"].unique())
    results = []

    for train_ds in datasets:
        for test_ds in datasets:
            train = df_clean[df_clean["dataset"] == train_ds]
            test = df_clean[df_clean["dataset"] == test_ds]
            if len(train) < 50 or len(test) < 10:
                continue

            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            model.fit(train[features].fillna(0), train[ztarget].values)
            pred = model.predict(test[features].fillna(0))
            y = test[ztarget].values

            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rho, _ = stats.spearmanr(pred, y)

            results.append({
                "train": train_ds, "test": test_ds,
                "r2": r2, "rho": rho,
                "n_train": len(train), "n_test": len(test),
            })
            tag = "IN-DOMAIN" if train_ds == test_ds else "CROSS"
            print(f"  {train_ds:4s} → {test_ds:4s}: R²(z)={r2:+.4f}  ρ={rho:.4f}  [{tag}]")

    # Plot heatmap
    rdf = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in [(axes[0], "r2", "R² (z-scored)"), (axes[1], "rho", "Spearman ρ")]:
        pivot = rdf.pivot_table(index="train", columns="test", values=col)
        cmap = "RdYlGn" if col == "r2" else "RdBu_r"
        vmin = -1 if col == "r2" else -1
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Test")
        ax.set_ylabel("Train")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=12)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    savefig(fig, os.path.join(outdir, "zscore_cross_dataset"))

    return rdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="psnr")
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    target = args.target
    df = zscore_normalize(df, target)
    ztarget = f"{target}_z"

    print(f"Loaded {len(df)} rows")
    print(f"Target: {ztarget} (z-score normalized {target})")
    print(f"Z-scored {target} stats: mean={df[ztarget].mean():.4f}, std={df[ztarget].std():.4f}")

    # 1. Correlations
    corr = run_correlations(df, ztarget, outdir)

    # 2. XGBoost regression
    xgb_results, best_model, best_features = run_xgboost(df, ztarget, outdir)

    # 3. SHAP
    shap_imp = run_shap(best_model, df, ztarget, best_features, outdir)

    # 4. Binary classification
    auc = run_binary_classification(df, ztarget, outdir)

    # 5. LOSO
    loso_df = run_loso(df, ztarget, outdir)

    # 6. Cross-dataset
    cross_df = run_cross_dataset(df, ztarget, outdir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (z-scored PSNR)")
    print("="*60)
    print(f"  Correlations: best = {label(corr[0]['metric'])} (ρ={corr[0]['spearman']:+.4f})"
          if corr else "")
    for name, res in xgb_results.items():
        print(f"  XGBoost [{name}]: R²={res['r2']}, ρ={res['rho']}")
    print(f"  Binary AUC: {auc:.4f}")
    print(f"  LOSO: R²(z)={loso_df['r2'].mean():.4f}, ρ={loso_df['rho'].mean():.4f}, "
          f"R²>0: {(loso_df['r2']>0).sum()}/{len(loso_df)}")
    in_domain = cross_df[cross_df["train"] == cross_df["test"]]
    cross_domain = cross_df[cross_df["train"] != cross_df["test"]]
    print(f"  Cross-dataset in-domain: R²(z)={in_domain['r2'].mean():.4f}")
    print(f"  Cross-dataset cross:     R²(z)={cross_domain['r2'].mean():.4f}")

    print(f"\nAll figures saved to {outdir}")


if __name__ == "__main__":
    main()

"""Merge per-frame fidelity and distance JSONs into a single master CSV.

Reads from results/perframe/{experiment}/{scene}_fidelity.json
         and results/perframe/{experiment}/{scene}_distances.json

Output: results/perframe/combined_perframe.csv

Usage:
    python merge_perframe_data.py
    python merge_perframe_data.py --experiments v3_k15_random v3_k15_infomax
"""

import os
import json
import argparse
import glob
import csv
from collections import defaultdict

WORKDIR = "/gpfs/workdir/malhotraa"
PERFRAME_BASE = f"{WORKDIR}/ConMax3D_reproduce/results/perframe"

# Map experiment names to method/budget
EXPERIMENT_MAP = {
    # k=10 experiments
    "v3": {"method": "infomax", "budget": 10},
    "v3_swap": {"method": "infomax_swap", "budget": 10},
    "v3_fvs_angular": {"method": "fvs_angular", "budget": 10},
    "v3_fvs_euclidean": {"method": "fvs_euclidean", "budget": 10},
    "v3_fvs_plucker": {"method": "fvs_plucker", "budget": 10},
    "v3_lpips": {"method": "lpips_fvs", "budget": 10},
    "v3_random_s42": {"method": "random", "budget": 10},
    "v3_random_s123": {"method": "random_s123", "budget": 10},
    "v3_random_s456": {"method": "random_s456", "budget": 10},
    # k=15 experiments
    "v3_k15_infomax": {"method": "infomax", "budget": 15},
    "v3_k15_fvs": {"method": "fvs", "budget": 15},
    "v3_k15_fvs_plucker": {"method": "fvs_plucker", "budget": 15},
    "v3_k15_random": {"method": "random", "budget": 15},
    # k=20 experiments
    "v3_k20_infomax": {"method": "infomax", "budget": 20},
    "v3_k20_fvs": {"method": "fvs", "budget": 20},
    "v3_k20_fvs_plucker": {"method": "fvs_plucker", "budget": 20},
    "v3_k20_random": {"method": "random", "budget": 20},
    # k=25 experiments
    "v3_k25_infomax": {"method": "infomax", "budget": 25},
    "v3_k25_fvs": {"method": "fvs", "budget": 25},
    "v3_k25_fvs_plucker": {"method": "fvs_plucker", "budget": 25},
    "v3_k25_random": {"method": "random", "budget": 25},
    # DINOv2 layer ablations (all k=10)
    "v3_dinov2_L2+4+6+8_concat": {"method": "infomax_L2468", "budget": 10},
    "v3_dinov2_L4+8_concat": {"method": "infomax_L48", "budget": 10},
    "v3_dinov2_L4+6+8+10_concat": {"method": "infomax_L46810", "budget": 10},
    "v3_dinov2_L2+4+6_concat": {"method": "infomax_L246", "budget": 10},
    "v3_dinov2_L0+2+4+6+8+10_concat": {"method": "infomax_L0246810", "budget": 10},
    "v3_dinov2_L0+1+2+3+4+5+6+7+8+9+10+11_concat": {"method": "infomax_Lall", "budget": 10},
    "v3_dinov2_L10": {"method": "infomax_L10", "budget": 10},
}

DISTANCE_COLUMNS = [
    "fvs_baseline", "fvs_plucker", "fvs_angular", "fvs_euclidean",
    "pc_max", "conmax3d_cov", "alexnet_entropy", "alexnet_dist",
    "dinov2_dist", "clip_dist", "infomax3d_marginal", "lpips_dist",
]

FIDELITY_COLUMNS = ["psnr", "ssim", "lpips"]

CSV_COLUMNS = [
    "dataset", "scene", "experiment", "method", "budget", "backend",
    "frame_id",
] + FIDELITY_COLUMNS + DISTANCE_COLUMNS


def parse_experiment(exp_name):
    """Parse experiment name into method and budget."""
    if exp_name in EXPERIMENT_MAP:
        info = EXPERIMENT_MAP[exp_name]
        return info["method"], info["budget"]

    # Try to infer from name
    parts = exp_name.split("_")
    method = "unknown"
    budget = 10

    for p in parts:
        if p.startswith("k") and p[1:].isdigit():
            budget = int(p[1:])

    if "random" in exp_name:
        method = "random"
    elif "fvs" in exp_name:
        if "plucker" in exp_name:
            method = "fvs_plucker"
        elif "angular" in exp_name:
            method = "fvs_angular"
        elif "euclidean" in exp_name:
            method = "fvs_euclidean"
        else:
            method = "fvs"
    elif "infomax" in exp_name or "dinov2" in exp_name:
        method = "infomax"
    elif "lpips" in exp_name:
        method = "lpips_fvs"

    return method, budget


def merge_all(experiments=None, output_path=None):
    """Merge all per-frame data into a single CSV."""
    if output_path is None:
        output_path = os.path.join(PERFRAME_BASE, "combined_perframe.csv")

    if experiments is None:
        # Discover all experiment directories
        exp_dirs = sorted(glob.glob(os.path.join(PERFRAME_BASE, "v3*")))
        experiments = [os.path.basename(d) for d in exp_dirs if os.path.isdir(d)]

    print(f"Processing {len(experiments)} experiments")
    rows = []

    for exp in experiments:
        exp_dir = os.path.join(PERFRAME_BASE, exp)
        if not os.path.isdir(exp_dir):
            print(f"  Skipping {exp}: directory not found")
            continue

        method, budget = parse_experiment(exp)

        # Find all scene fidelity files
        fidelity_files = sorted(glob.glob(os.path.join(exp_dir, "*_fidelity.json")))
        for fid_file in fidelity_files:
            scene = os.path.basename(fid_file).replace("_fidelity.json", "")
            dist_file = os.path.join(exp_dir, f"{scene}_distances.json")

            # Load fidelity data
            with open(fid_file) as f:
                fid_data = json.load(f)

            dataset = fid_data.get("dataset", "unknown")
            backend = "3DGS"

            # Build frame_id -> fidelity map
            fid_map = {}
            for frame in fid_data.get("per_frame", []):
                fid_map[frame["frame_id"]] = frame

            # Load distance data if available
            dist_map = {}
            if os.path.exists(dist_file):
                with open(dist_file) as f:
                    dist_data = json.load(f)
                for frame in dist_data.get("per_frame", []):
                    dist_map[frame["frame_id"]] = frame

            # Merge
            all_frame_ids = sorted(set(list(fid_map.keys()) + list(dist_map.keys())))
            for fid in all_frame_ids:
                row = {
                    "dataset": dataset,
                    "scene": scene,
                    "experiment": exp,
                    "method": method,
                    "budget": budget,
                    "backend": backend,
                    "frame_id": fid,
                }

                # Add fidelity columns
                fid_frame = fid_map.get(fid, {})
                for col in FIDELITY_COLUMNS:
                    row[col] = fid_frame.get(col, "")

                # Add distance columns
                dist_frame = dist_map.get(fid, {})
                for col in DISTANCE_COLUMNS:
                    row[col] = dist_frame.get(col, "")

                rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWritten {len(rows)} rows to {output_path}")

    # Summary statistics
    datasets = set(r["dataset"] for r in rows)
    scenes = set(r["scene"] for r in rows)
    methods = set(r["method"] for r in rows)
    print(f"  Datasets: {sorted(datasets)}")
    print(f"  Scenes: {len(scenes)}")
    print(f"  Methods: {sorted(methods)}")

    # Check completeness
    n_with_fidelity = sum(1 for r in rows if r.get("psnr"))
    n_with_distances = sum(1 for r in rows if r.get("fvs_baseline"))
    n_complete = sum(1 for r in rows if r.get("psnr") and r.get("fvs_baseline"))
    print(f"  With fidelity: {n_with_fidelity}")
    print(f"  With distances: {n_with_distances}")
    print(f"  Complete (both): {n_complete}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Merge per-frame data into master CSV")
    parser.add_argument("--experiments", nargs="+", help="Specific experiments to merge")
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    merge_all(experiments=args.experiments, output_path=args.output)


if __name__ == "__main__":
    main()

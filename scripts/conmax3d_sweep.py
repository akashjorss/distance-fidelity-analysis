"""W&B sweep wrapper for ConMax3D hyperparameter tuning.

Supports per-dataset sweeps via SWEEP_DATASET env var.
Each trial:
1. Gets hyperparams from W&B sweep agent
2. Runs ConMax3D frame selection with those params on ~3 scenes
3. Trains a quick 3DGS model with selected frames
4. Logs PSNR to W&B for the sweep to optimize

Usage:
  export SWEEP_DATASET=llff  # or tt, ns
  wandb sweep sweep_config_bayesian.yaml
  wandb agent <sweep_id>
"""

import os
import sys
import json
import subprocess
import wandb

WORKDIR = "/gpfs/workdir/malhotraa"
OUTPUT_DIR = WORKDIR + "/ConMax3D_reproduce/results/sweep"
SCRIPTS_DIR = WORKDIR + "/ConMax3D_reproduce/scripts"
GSPLAT_DIR = WORKDIR + "/gsplat/examples"
GSPLAT_PYTHON = WORKDIR + "/conda_envs/gsplat_env/bin/python"

DATASET_CONFIGS = {
    "llff": {
        "scenes": ["fern", "fortress", "trex"],
        "data_dir": WORKDIR + "/data/LLFF",
        "gsplat_dataset_type": "colmap",
        "conmax3d_dataset_type": "llff",
        "data_factor": 4,
        "init_type": "sfm",
    },
    "tt": {
        "scenes": ["Church", "Barn", "Museum"],
        "data_dir": WORKDIR + "/data/Tanks",
        "gsplat_dataset_type": "colmap",
        "conmax3d_dataset_type": "llff",
        "data_factor": 1,
        "init_type": "sfm",
    },
    "ns": {
        "scenes": ["chair", "lego", "materials"],
        "data_dir": WORKDIR + "/data/nerf_synthetic_gsplat",
        "gsplat_dataset_type": "nerfstyle",
        "conmax3d_dataset_type": "nerf_synthetic",
        "data_factor": 1,
        "init_type": "random",
    },
}

SWEEP_K = 10
GSPLAT_STEPS = 10000


def get_gsplat_env():
    """Get a clean environment for gsplat subprocess."""
    env = os.environ.copy()
    # Remove PYTHONPATH entries that conflict with gsplat_env's installed packages
    if "PYTHONPATH" in env:
        paths = env["PYTHONPATH"].split(":")
        clean_paths = [p for p in paths if "gsplat" not in p]
        env["PYTHONPATH"] = ":".join(clean_paths) if clean_paths else ""
    # Add gsplat_env bin to PATH for ninja and other tools
    gsplat_bin = WORKDIR + "/conda_envs/gsplat_env/bin"
    if gsplat_bin not in env.get("PATH", ""):
        env["PATH"] = gsplat_bin + ":" + env.get("PATH", "")
    return env


def get_config():
    dataset = os.environ.get("SWEEP_DATASET", "llff")
    if dataset not in DATASET_CONFIGS:
        raise ValueError("Unknown dataset: " + dataset)
    return dataset, DATASET_CONFIGS[dataset]


def run_conmax3d(scene, config, cfg):
    """Run ConMax3D frame selection and return selected indices."""
    cmd = [
        sys.executable,
        SCRIPTS_DIR + "/conmax3d_sam2_wandb.py",
        "--base_dir", cfg["data_dir"],
        "--scene", scene,
        "--output_dir", OUTPUT_DIR + "/" + wandb.run.id,
        "--num_frames", str(SWEEP_K),
        "--dataset_type", cfg["conmax3d_dataset_type"],
        "--pred_iou_thresh", str(config["pred_iou_thresh"]),
        "--min_cluster_factor", str(config["min_cluster_factor"]),
        "--efficientnet_model", config["efficientnet_model"],
        "--downscale_factor", str(config["downscale_factor"]),
        "--sam2_checkpoint", WORKDIR + "/segment-anything-2/checkpoints/sam2_hiera_large.pt",
    ]
    print("Running: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR: " + result.stderr[-1000:])
        return None

    k_word = "ten"
    indices_file = OUTPUT_DIR + "/" + wandb.run.id + "/" + k_word + "/conmax3d/train_indices.json"
    if os.path.exists(indices_file):
        with open(indices_file) as f:
            data = json.load(f)
        return data.get(scene)
    return None


def train_gsplat_quick(scene, train_indices, cfg):
    """Quick 3DGS training and return PSNR."""
    indices_str = ",".join(map(str, train_indices))
    result_dir = OUTPUT_DIR + "/" + wandb.run.id + "/gsplat/" + scene
    scene_data_dir = cfg["data_dir"] + "/" + scene

    cmd = [
        GSPLAT_PYTHON,
        GSPLAT_DIR + "/simple_trainer.py",
        "default",
        "--data_dir", scene_data_dir,
        "--dataset_type", cfg["gsplat_dataset_type"],
        "--data_factor", str(cfg["data_factor"]),
        "--init_type", cfg["init_type"],
        "--train_indices", indices_str,
        "--result_dir", result_dir,
        "--max_steps", str(GSPLAT_STEPS),
        "--eval_steps", str(GSPLAT_STEPS),
        "--save_steps", str(GSPLAT_STEPS),
        "--disable_viewer",
    ]
    print("Training 3DGS: " + " ".join(cmd))
    gsplat_env = get_gsplat_env()
    result = subprocess.run(cmd, capture_output=True, text=True, env=gsplat_env)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR: " + result.stderr[-1000:])
        return None

    stats_dir = os.path.join(result_dir, "stats")
    if os.path.exists(stats_dir):
        for f in os.listdir(stats_dir):
            if f.startswith("val_step") and f.endswith(".json"):
                with open(os.path.join(stats_dir, f)) as fh:
                    stats = json.load(fh)
                return stats.get("psnr")
    return None


def main():
    dataset_name, cfg = get_config()

    wandb.init(tags=[dataset_name])
    config = dict(wandb.config)
    print("Dataset: " + dataset_name + ", Scenes: " + str(cfg["scenes"]))
    print("Sweep config: " + str(config))

    psnr_values = []
    for scene in cfg["scenes"]:
        print("\n=== Scene: " + scene + " ===")
        indices = run_conmax3d(scene, config, cfg)
        if indices is None:
            print("Failed to get indices for " + scene)
            wandb.log({scene + "_psnr": 0})
            continue

        psnr = train_gsplat_quick(scene, indices, cfg)
        if psnr is not None:
            psnr_values.append(psnr)
            wandb.log({scene + "_psnr": psnr})
            print(scene + " PSNR: " + str(round(psnr, 2)))
        else:
            print("Failed to get PSNR for " + scene)
            wandb.log({scene + "_psnr": 0})

    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    wandb.log({"avg_psnr": avg_psnr})
    print("\nAverage PSNR: " + str(round(avg_psnr, 2)))
    wandb.finish()


if __name__ == "__main__":
    main()

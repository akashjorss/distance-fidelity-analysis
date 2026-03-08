"""Extract per-frame fidelity metrics (PSNR/SSIM/LPIPS) from trained gsplat checkpoints.

For each experiment/scene, loads the gsplat checkpoint, renders each test view,
and saves per-frame metrics to JSON.

Usage:
    python extract_perframe_fidelity.py --experiment v3_k15_random --scene fern
    python extract_perframe_fidelity.py --experiment v3_k15_random  # all scenes
    python extract_perframe_fidelity.py --list  # list available experiments
"""

import os
import sys
import json
import argparse
import math
import glob
import numpy as np
import torch
from pathlib import Path

WORKDIR = "/gpfs/workdir/malhotraa"
RESULTS_BASE = f"{WORKDIR}/ConMax3D_reproduce/results"
GSPLAT_DIR = f"{WORKDIR}/gsplat/examples"
OUTPUT_BASE = f"{RESULTS_BASE}/perframe"

# Add gsplat examples to path for importing simple_trainer utilities
sys.path.insert(0, GSPLAT_DIR)

LLFF_SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
TT_SCENES = ["Ballroom", "Barn", "Church", "Family", "Francis", "Horse", "Ignatius", "Museum"]
NS_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

DATASET_CONFIGS = {
    "llff": {
        "scenes": LLFF_SCENES,
        "data_dir": f"{WORKDIR}/data/LLFF",
        "dataset_type": "colmap",
        "data_factor": 4,
        "init_type": "sfm",
    },
    "tt": {
        "scenes": TT_SCENES,
        "data_dir": f"{WORKDIR}/data/Tanks",
        "dataset_type": "colmap",
        "data_factor": 1,
        "init_type": "sfm",
    },
    "ns": {
        "scenes": NS_SCENES,
        "data_dir": f"{WORKDIR}/data/nerf_synthetic_gsplat",
        "dataset_type": "nerfstyle",
        "data_factor": 1,
        "init_type": "random",
    },
}


def get_dataset_for_scene(scene):
    """Determine which dataset a scene belongs to."""
    if scene in LLFF_SCENES:
        return "llff"
    elif scene in TT_SCENES:
        return "tt"
    elif scene in NS_SCENES:
        return "ns"
    return None


def find_checkpoint(experiment, scene):
    """Find the gsplat checkpoint for a given experiment and scene."""
    exp_dir = os.path.join(RESULTS_BASE, experiment, scene)
    if not os.path.isdir(exp_dir):
        return None, None

    # Find gsplat_* subdirectory
    gsplat_dirs = glob.glob(os.path.join(exp_dir, "gsplat_*"))
    if not gsplat_dirs:
        return None, None

    for gdir in gsplat_dirs:
        ckpt_path = os.path.join(gdir, "ckpts", "ckpt_29999_rank0.pt")
        if os.path.exists(ckpt_path):
            # Extract method name from gsplat_<method>
            method = os.path.basename(gdir).replace("gsplat_", "")
            return ckpt_path, method

    return None, None


def find_train_indices(experiment, scene):
    """Find train indices JSON for a given experiment and scene."""
    exp_dir = os.path.join(RESULTS_BASE, experiment, scene)
    pattern = os.path.join(exp_dir, f"train_indices_{scene}_*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        return data.get("selected_indices", [])
    return None


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two images (simplified)."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(img1.device)
    # Add batch dimension if needed: [C, H, W] -> [1, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    return ssim_metric(img1, img2).item()


def compute_lpips(img1, img2, lpips_fn):
    """Compute LPIPS between two images."""
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    return lpips_fn(img1, img2).item()


def extract_perframe(experiment, scene, device="cuda"):
    """Extract per-frame fidelity metrics for a single experiment/scene pair."""
    dataset_name = get_dataset_for_scene(scene)
    if dataset_name is None:
        print(f"Unknown scene: {scene}")
        return None

    cfg = DATASET_CONFIGS[dataset_name]
    ckpt_path, method = find_checkpoint(experiment, scene)
    if ckpt_path is None:
        print(f"No checkpoint found for {experiment}/{scene}")
        return None

    train_indices = find_train_indices(experiment, scene)
    if train_indices is None:
        print(f"No train indices found for {experiment}/{scene}")
        return None

    print(f"Processing {experiment}/{scene} (method={method}, dataset={dataset_name})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Train indices: {train_indices}")

    # Import gsplat components
    from simple_trainer import Runner, Config

    # Build config matching the training setup
    cli_args = [
        "default",
        "--data_dir", os.path.join(cfg["data_dir"], scene),
        "--dataset_type", cfg["dataset_type"],
        "--data_factor", str(cfg["data_factor"]),
        "--init_type", cfg["init_type"],
        "--train_indices", ",".join(map(str, train_indices)),
        "--result_dir", "/tmp/gsplat_eval_dummy",
        "--max_steps", "1",
        "--eval_steps", "1",
        "--disable_viewer",
    ]

    # Parse config
    cfg_obj = Config()
    import tyro
    cfg_obj = tyro.cli(Config, args=cli_args)

    # Create runner and load checkpoint
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg_obj)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    # The checkpoint contains splat parameters
    for k, v in ckpt.items():
        if hasattr(runner, "splats") and k in runner.splats:
            runner.splats[k] = v.to(device)
        elif k == "splats":
            for sk, sv in v.items():
                runner.splats[sk] = sv.to(device)

    # Setup LPIPS
    import lpips as lpips_module
    lpips_fn = lpips_module.LPIPS(net="alex").to(device)

    # Get test dataset
    test_dataset = runner.valset
    n_test = len(test_dataset)
    print(f"  Test frames: {n_test}")

    results = []
    runner.eval()

    with torch.no_grad():
        for i in range(n_test):
            data = test_dataset[i]
            camtoworlds = data["camtoworld"][None].to(device)  # [1, 4, 4]
            Ks = data["K"][None].to(device)  # [1, 3, 3]
            width = data["width"]
            height = data["height"]
            gt_image = data["image"].to(device)  # [H, W, 3]

            # Render
            renders, _, _ = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
            )
            rendered = renders[0]  # [H, W, 3]

            # Clamp to [0, 1]
            rendered = torch.clamp(rendered, 0.0, 1.0)

            # Compute metrics - convert to [C, H, W] for SSIM/LPIPS
            gt_chw = gt_image.permute(2, 0, 1)
            rendered_chw = rendered.permute(2, 0, 1)

            psnr = compute_psnr(rendered, gt_image)
            ssim = compute_ssim(rendered_chw, gt_chw)
            lpips_val = compute_lpips(
                rendered_chw * 2 - 1,  # LPIPS expects [-1, 1]
                gt_chw * 2 - 1,
                lpips_fn,
            )

            frame_result = {
                "frame_id": i,
                "psnr": round(psnr, 6),
                "ssim": round(ssim, 6),
                "lpips": round(lpips_val, 6),
            }
            results.append(frame_result)

            if (i + 1) % 10 == 0 or i == n_test - 1:
                print(f"  Frame {i+1}/{n_test}: PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips_val:.4f}")

    # Compute averages for sanity check
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_lpips = np.mean([r["lpips"] for r in results])

    output = {
        "experiment": experiment,
        "scene": scene,
        "dataset": dataset_name,
        "method": method,
        "train_indices": train_indices,
        "n_test_frames": n_test,
        "avg_psnr": round(float(avg_psnr), 6),
        "avg_ssim": round(float(avg_ssim), 6),
        "avg_lpips": round(float(avg_lpips), 6),
        "per_frame": results,
    }

    # Save output
    out_dir = os.path.join(OUTPUT_BASE, experiment)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}_fidelity.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {out_path}")
    print(f"  Averages: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")

    return output


def list_experiments():
    """List available experiments with checkpoints."""
    exp_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, "v3_*")))
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        # Count scenes with checkpoints
        ckpts = glob.glob(os.path.join(exp_dir, "*/gsplat_*/ckpts/ckpt_29999_rank0.pt"))
        scenes = [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))) for c in ckpts]
        if scenes:
            print(f"{exp_name}: {len(scenes)} scenes ({', '.join(sorted(scenes)[:4])}...)")


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame fidelity from gsplat checkpoints")
    parser.add_argument("--experiment", type=str, help="Experiment directory name (e.g., v3_k15_random)")
    parser.add_argument("--scene", type=str, help="Scene name (if omitted, process all scenes)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list)")

    if args.scene:
        scenes = [args.scene]
    else:
        # Process all scenes that have checkpoints for this experiment
        exp_dir = os.path.join(RESULTS_BASE, args.experiment)
        ckpts = glob.glob(os.path.join(exp_dir, "*/gsplat_*/ckpts/ckpt_29999_rank0.pt"))
        scenes = sorted(set(
            os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c))))
            for c in ckpts
        ))
        print(f"Found {len(scenes)} scenes with checkpoints: {scenes}")

    for scene in scenes:
        try:
            extract_perframe(args.experiment, scene, device=args.device)
        except Exception as e:
            print(f"ERROR processing {args.experiment}/{scene}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

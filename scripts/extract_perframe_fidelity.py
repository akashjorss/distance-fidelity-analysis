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
import glob
import time
import numpy as np
import torch

WORKDIR = "/gpfs/workdir/malhotraa"
RESULTS_BASE = f"{WORKDIR}/ConMax3D_reproduce/results"
GSPLAT_DIR = f"{WORKDIR}/gsplat/examples"
OUTPUT_BASE = f"{RESULTS_BASE}/perframe"

# Add gsplat examples to path
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
    if scene in LLFF_SCENES:
        return "llff"
    elif scene in TT_SCENES:
        return "tt"
    elif scene in NS_SCENES:
        return "ns"
    return None


def find_checkpoint(experiment, scene):
    exp_dir = os.path.join(RESULTS_BASE, experiment, scene)
    if not os.path.isdir(exp_dir):
        return None, None
    gsplat_dirs = glob.glob(os.path.join(exp_dir, "gsplat_*"))
    for gdir in gsplat_dirs:
        ckpt_path = os.path.join(gdir, "ckpts", "ckpt_29999_rank0.pt")
        if os.path.exists(ckpt_path):
            method = os.path.basename(gdir).replace("gsplat_", "")
            return ckpt_path, method
    return None, None


def find_train_indices(experiment, scene):
    exp_dir = os.path.join(RESULTS_BASE, experiment, scene)
    pattern = os.path.join(exp_dir, f"train_indices_{scene}_*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        return data.get("selected_indices", [])
    return None


def extract_perframe(experiment, scene, device="cuda"):
    """Extract per-frame fidelity metrics using gsplat's Runner."""
    dataset_name = get_dataset_for_scene(scene)
    if dataset_name is None:
        print(f"Unknown scene: {scene}")
        return None

    cfg_dict = DATASET_CONFIGS[dataset_name]
    ckpt_path, method = find_checkpoint(experiment, scene)
    if ckpt_path is None:
        print(f"No checkpoint found for {experiment}/{scene}")
        return None

    train_indices = find_train_indices(experiment, scene)
    if train_indices is None:
        print(f"No train indices found for {experiment}/{scene}")
        return None

    # Check if output already exists
    out_dir = os.path.join(OUTPUT_BASE, experiment)
    out_path = os.path.join(out_dir, f"{scene}_fidelity.json")
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}, skipping")
        return None

    print(f"Processing {experiment}/{scene} (method={method}, dataset={dataset_name})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Train indices ({len(train_indices)}): {train_indices}")

    # Build Config and Runner using gsplat's API
    from simple_trainer import Config, Runner

    # Create config via dataclass constructor (avoid tyro CLI parsing)
    config = Config()
    config.data_dir = os.path.join(cfg_dict["data_dir"], scene)
    config.data_factor = cfg_dict["data_factor"]
    config.init_type = cfg_dict["init_type"]
    config.disable_viewer = True
    config.result_dir = os.path.join("/tmp", f"gsplat_eval_{experiment}_{scene}")
    config.max_steps = 1
    config.eval_steps = [1]
    config.save_steps = []

    # Create runner with train indices
    train_indices_list = [int(i) for i in train_indices]
    runner = Runner(
        local_rank=0, world_rank=0, world_size=1,
        cfg=config, train_indices=train_indices_list,
    )

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    for k in runner.splats.keys():
        if k in ckpt.get("splats", {}):
            runner.splats[k].data = ckpt["splats"][k]
    print(f"  Loaded checkpoint: {len(runner.splats['means'])} Gaussians")

    # Run per-frame evaluation
    valloader = torch.utils.data.DataLoader(
        runner.valset, batch_size=1, shuffle=False, num_workers=1
    )

    results = []
    n_test = len(runner.valset)
    print(f"  Test frames: {n_test}")

    with torch.no_grad():
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(runner.device)
            Ks = data["K"].to(runner.device)
            pixels = data["image"].to(runner.device) / 255.0
            height, width = pixels.shape[1:3]

            colors, _, _ = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=config.sh_degree,
                near_plane=config.near_plane,
                far_plane=config.far_plane,
            )
            colors = torch.clamp(colors, 0.0, 1.0)

            # [1, H, W, 3] -> [1, 3, H, W] for metrics
            pixels_chw = pixels.permute(0, 3, 1, 2)
            colors_chw = colors.permute(0, 3, 1, 2)

            psnr = runner.psnr(colors_chw, pixels_chw).item()
            ssim = runner.ssim(colors_chw, pixels_chw).item()
            lpips_val = runner.lpips(colors_chw, pixels_chw).item()

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
        "n_train": len(train_indices),
        "n_test_frames": n_test,
        "avg_psnr": round(float(avg_psnr), 6),
        "avg_ssim": round(float(avg_ssim), 6),
        "avg_lpips": round(float(avg_lpips), 6),
        "per_frame": results,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to: {out_path}")
    print(f"  Averages: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}")

    # Cleanup temp dir
    import shutil
    if os.path.exists(config.result_dir):
        shutil.rmtree(config.result_dir, ignore_errors=True)

    return output


def list_experiments():
    exp_dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, "v3_*")))
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        ckpts = glob.glob(os.path.join(exp_dir, "*/gsplat_*/ckpts/ckpt_29999_rank0.pt"))
        scenes = [os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(c)))) for c in ckpts]
        if scenes:
            print(f"{exp_name}: {len(scenes)} scenes ({', '.join(sorted(scenes)[:4])}...)")


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame fidelity from gsplat checkpoints")
    parser.add_argument("--experiment", type=str, help="Experiment directory name")
    parser.add_argument("--scene", type=str, help="Scene name (if omitted, process all)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list)")

    if args.scene:
        scenes = [args.scene]
    else:
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

"""Extract per-frame fidelity metrics from NeRF rendered test images.

Compares rendered PNGs against GT images loaded from the dataset.
Outputs JSON in same format as extract_perframe_fidelity.py for merging.

Usage:
    python scripts/extract_nerf_perframe.py
    python scripts/extract_nerf_perframe.py --method fvs_euclidean --scene fern
"""
import os, sys, json, argparse, glob
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

WORKDIR = "/gpfs/workdir/malhotraa"
RESULTS_BASE = f"{WORKDIR}/ConMax3D_reproduce/results"
OUTPUT_BASE = f"{RESULTS_BASE}/perframe"

LLFF_SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
TT_SCENES = ["Ballroom", "Barn", "Church", "Family", "Francis", "Horse", "Ignatius", "Museum"]

NERF_METHODS = [
    "fvs_euclidean", "fvs_angular", "fvs_plucker",
    "infomax", "fvs", "lpips_fvs", "random_s42",
]

# Map NeRF method to gsplat results dir (for train indices)
METHOD_TO_GSPLAT = {
    "fvs_euclidean": "v3_fvs_euclidean",
    "fvs_angular": "v3_fvs_angular",
    "fvs_plucker": "v3_fvs_plucker",
    "infomax": "v3_dinov2_L4",
    "fvs": "v3",
    "lpips_fvs": "v3_lpips",
    "random_s42": "v3_random_s42",
}

# Map NeRF method to strategy name in index JSON
METHOD_TO_STRAT = {
    "fvs_euclidean": "fvs_euclidean",
    "fvs_angular": "fvs_angular",
    "fvs_plucker": "fvs_plucker",
    "infomax": "infomax",
    "fvs": "fvs",
    "lpips_fvs": "lpips_fvs",
    "random_s42": "random",
}

DATASET_CONFIGS = {
    "llff": {"data_dir": f"{WORKDIR}/data/LLFF", "factor": 4, "img_subdir": "images_4"},
    "tt": {"data_dir": f"{WORKDIR}/data/Tanks", "factor": 1, "img_subdir": "images"},
}


def get_dataset(scene):
    if scene in LLFF_SCENES:
        return "llff"
    elif scene in TT_SCENES:
        return "tt"
    return None


def load_gt_images(scene, dataset):
    """Load all GT images for a scene, return as sorted list of (global_idx, image_array)."""
    cfg = DATASET_CONFIGS[dataset]
    img_dir = os.path.join(cfg["data_dir"], scene, cfg["img_subdir"])
    if not os.path.isdir(img_dir):
        # Try 'images' directly
        img_dir = os.path.join(cfg["data_dir"], scene, "images")
    if not os.path.isdir(img_dir):
        return None

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.jpg")) +
                       glob.glob(os.path.join(img_dir, "*.JPG")))
    return img_files


def get_train_indices(method, scene):
    """Load train indices from the gsplat experiment directory."""
    gsplat_exp = METHOD_TO_GSPLAT.get(method)
    strat = METHOD_TO_STRAT.get(method)
    if not gsplat_exp or not strat:
        return None

    idx_file = os.path.join(RESULTS_BASE, gsplat_exp, scene, f"train_indices_{scene}_{strat}.json")
    if not os.path.exists(idx_file):
        return None

    with open(idx_file) as f:
        data = json.load(f)
    return data.get("selected_indices", [])


def compute_perframe(method, scene):
    """Compute per-frame PSNR/SSIM from NeRF renderings vs GT."""
    dataset = get_dataset(scene)
    if not dataset:
        return None

    nerf_dir = os.path.join(RESULTS_BASE, f"v3_nerf_{method}", scene,
                            f"{scene}_{method}", "testset_050000")
    if not os.path.isdir(nerf_dir):
        return None

    train_indices = get_train_indices(method, scene)
    if train_indices is None:
        print(f"  No train indices for {method}/{scene}")
        return None

    gt_files = load_gt_images(scene, dataset)
    if not gt_files:
        print(f"  No GT images for {scene}")
        return None

    n_total = len(gt_files)
    test_indices = sorted([i for i in range(n_total) if i not in train_indices])

    # Check rendered files exist
    rendered_files = sorted(glob.glob(os.path.join(nerf_dir, "*.png")))
    if len(rendered_files) != len(test_indices):
        print(f"  Mismatch: {len(rendered_files)} rendered vs {len(test_indices)} test indices")
        # Try to match anyway
        if len(rendered_files) == 0:
            return None

    results = []
    for seq_idx, global_idx in enumerate(test_indices):
        rendered_path = os.path.join(nerf_dir, f"{seq_idx:03d}.png")
        if not os.path.exists(rendered_path):
            continue

        gt_path = gt_files[global_idx]

        rendered = np.array(Image.open(rendered_path)).astype(np.float64) / 255.0
        gt = np.array(Image.open(gt_path)).astype(np.float64) / 255.0

        # Resize if needed (NeRF may render at different resolution)
        if rendered.shape != gt.shape:
            from PIL import Image as PILImage
            gt_pil = PILImage.open(gt_path).resize(
                (rendered.shape[1], rendered.shape[0]), PILImage.LANCZOS)
            gt = np.array(gt_pil).astype(np.float64) / 255.0

        # Ensure same number of channels
        if rendered.ndim == 2:
            rendered = np.stack([rendered]*3, axis=-1)
        if gt.ndim == 2:
            gt = np.stack([gt]*3, axis=-1)
        if rendered.shape[-1] == 4:
            rendered = rendered[:, :, :3]
        if gt.shape[-1] == 4:
            gt = gt[:, :, :3]

        psnr = psnr_fn(gt, rendered, data_range=1.0)
        ssim_val = ssim_fn(gt, rendered, data_range=1.0, channel_axis=-1)

        results.append({
            "frame_id": global_idx,
            "valset_idx": seq_idx,
            "psnr": round(float(psnr), 6),
            "ssim": round(float(ssim_val), 6),
            "lpips": 0.0,  # Skip LPIPS for CPU-only computation
        })

    if not results:
        return None

    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])

    output = {
        "experiment": f"v3_nerf_{method}",
        "scene": scene,
        "dataset": dataset,
        "method": method,
        "backend": "NeRF",
        "train_indices": train_indices,
        "n_train": len(train_indices),
        "n_test_frames": len(results),
        "avg_psnr": round(float(avg_psnr), 6),
        "avg_ssim": round(float(avg_ssim), 6),
        "per_frame": results,
    }

    # Save
    out_dir = os.path.join(OUTPUT_BASE, f"v3_nerf_{method}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}_fidelity.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({len(results)} frames, PSNR={avg_psnr:.2f})")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="NeRF method")
    parser.add_argument("--scene", type=str, help="Scene name")
    args = parser.parse_args()

    methods = [args.method] if args.method else NERF_METHODS
    all_scenes = LLFF_SCENES + TT_SCENES

    for method in methods:
        scenes = [args.scene] if args.scene else all_scenes
        print(f"\n=== NeRF method: {method} ===")
        for scene in scenes:
            try:
                compute_perframe(method, scene)
            except Exception as e:
                print(f"  ERROR {method}/{scene}: {e}")


if __name__ == "__main__":
    main()

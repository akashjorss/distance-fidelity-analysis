"""Preprocess NeRF Synthetic scenes with DuSt3R to generate point clouds.

Runs DuSt3R on a subset of training images, aligns the reconstructed
point cloud to the GT world frame via Sim(3) (Umeyama algorithm),
and saves the result as dust3r_points.npz for gsplat sfm init.

Usage:
  python scripts/preprocess_dust3r_ns.py \
    --data_dir $WORKDIR/data/nerf_synthetic_gsplat/chair \
    --dust3r_path $WORKDIR/stable-virtual-camera/third_party/dust3r \
    --seva_path $WORKDIR/stable-virtual-camera \
    --num_images 30 \
    --conf_threshold 1.5
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def umeyama_sim3(src, dst):
    """Compute Sim(3) alignment: dst ~ s * R @ src + t.

    Args:
        src: (N, 3) source points
        dst: (N, 3) target points

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert src.shape == dst.shape
    n = len(src)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    var_s = (src_c ** 2).sum() / n
    cov = dst_c.T @ src_c / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s

    return s, R, t


def load_gt_poses(data_dir):
    """Load GT camera-to-world poses from transforms.json."""
    json_path = os.path.join(data_dir, "transforms.json")
    with open(json_path) as f:
        meta = json.load(f)

    c2ws = []
    file_paths = []
    for frame in meta["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w_4x4 = np.eye(4, dtype=np.float64)
            c2w_4x4[:3, :] = c2w
            c2w = c2w_4x4
        c2ws.append(c2w)
        file_paths.append(frame["file_path"])

    return np.stack(c2ws, axis=0), file_paths


def main():
    parser = argparse.ArgumentParser(description="DuSt3R preprocessing for NeRF Synthetic")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to scene dir (contains transforms.json and images/)")
    parser.add_argument("--dust3r_path", type=str, required=True,
                        help="Path to dust3r source")
    parser.add_argument("--seva_path", type=str, required=True,
                        help="Path to stable-virtual-camera repo")
    parser.add_argument("--num_images", type=int, default=30,
                        help="Number of images to use for DuSt3R (uniformly sampled)")
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="Minimum DuSt3R confidence to keep a point")
    parser.add_argument("--output", type=str, default="dust3r_points.npz",
                        help="Output filename")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="DuSt3R inference batch size")
    parser.add_argument("--niter", type=int, default=500,
                        help="DuSt3R global alignment iterations")
    args = parser.parse_args()

    t0 = time.time()

    # Setup paths for DuSt3R imports
    sys.path.insert(0, args.seva_path)
    sys.path.insert(0, args.dust3r_path)

    from seva.modules.preprocessor import Dust3rPipeline

    # Load GT poses and image paths
    gt_c2ws, file_paths = load_gt_poses(args.data_dir)
    n_total = len(gt_c2ws)
    print(f"Scene: {os.path.basename(args.data_dir)}")
    print(f"Total training images: {n_total}")

    # Uniformly sample images for DuSt3R
    n_use = min(args.num_images, n_total)
    sample_indices = np.linspace(0, n_total - 1, n_use, dtype=int)
    print(f"Using {n_use} images for DuSt3R reconstruction")

    # Build image paths
    img_paths = []
    for idx in sample_indices:
        fp = file_paths[idx]
        full_path = os.path.join(args.data_dir, fp)
        if not os.path.exists(full_path):
            # Try without leading ./
            if fp.startswith("./"):
                fp = fp[2:]
            full_path = os.path.join(args.data_dir, fp)
        assert os.path.exists(full_path), f"Image not found: {full_path}"
        img_paths.append(full_path)

    gt_c2ws_sampled = gt_c2ws[sample_indices]

    # Run DuSt3R
    print("Initializing DuSt3R pipeline...")
    pipeline = Dust3rPipeline(device="cuda:0")

    print("Running DuSt3R inference and global alignment...")
    imgs, Ks, dust3r_c2ws, points_list, colors_list, depthmaps, conf_masks = \
        pipeline.infer_cameras_and_points(
            img_paths,
            batch_size=args.batch_size,
            niter=args.niter,
        )
    print(f"DuSt3R returned {len(points_list)} point clouds")

    # Extract camera positions for alignment
    # DuSt3R c2ws are in OpenCV convention, GT c2ws are in OpenGL convention
    # But camera positions (c2w[:3, 3]) are the same regardless of convention
    t_dust3r = dust3r_c2ws[:, :3, 3]  # (N, 3)
    t_gt = gt_c2ws_sampled[:, :3, 3]  # (N, 3)

    print(f"DuSt3R camera positions range: {t_dust3r.min(0)} to {t_dust3r.max(0)}")
    print(f"GT camera positions range: {t_gt.min(0)} to {t_gt.max(0)}")

    # Compute Sim(3) alignment
    s, R, t = umeyama_sim3(t_dust3r, t_gt)
    print(f"Sim(3) alignment: scale={s:.4f}")

    # Verify alignment quality
    t_aligned = s * (R @ t_dust3r.T).T + t
    alignment_error = np.linalg.norm(t_aligned - t_gt, axis=1)
    print(f"Alignment error: mean={alignment_error.mean():.4f}, max={alignment_error.max():.4f}")

    # Concatenate all point clouds and apply Sim(3)
    all_points = np.concatenate(points_list, axis=0)  # (M, 3)
    all_colors = np.concatenate(colors_list, axis=0)   # (M, 3)
    print(f"Total points before filtering: {len(all_points)}")

    # Apply Sim(3) transform to points
    all_points_aligned = s * (R @ all_points.T).T + t

    # Convert colors to [0, 255] range if needed
    if all_colors.max() <= 1.0:
        all_colors = (all_colors * 255).astype(np.float32)
    else:
        all_colors = all_colors.astype(np.float32)

    # Filter by scene bounds (remove obvious outliers)
    # Use GT camera positions to estimate scene bounds
    cam_center = t_gt.mean(0)
    cam_spread = np.linalg.norm(t_gt - cam_center, axis=1).max()
    bound = cam_spread * 5  # generous bound
    mask = np.linalg.norm(all_points_aligned - cam_center, axis=1) < bound
    all_points_aligned = all_points_aligned[mask]
    all_colors = all_colors[mask]
    print(f"Points after bound filtering: {len(all_points_aligned)}")

    # Save
    output_path = os.path.join(args.data_dir, args.output)
    np.savez(
        output_path,
        points=all_points_aligned.astype(np.float32),
        colors=all_colors.astype(np.float32),
    )
    elapsed = time.time() - t0
    print(f"Saved {len(all_points_aligned)} points to {output_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

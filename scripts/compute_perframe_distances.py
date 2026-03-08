"""Compute per-frame distance metrics between each test frame and the training set.

For each (experiment, scene), computes distances under multiple metrics:
  1. FVS baseline (min Euclidean camera center distance)
  2. FVS-Plucker (min 6D Plucker distance)
  3. FVS-Angular (min geodesic angular distance)
  4. FVS-Euclidean (min translation distance)
  5. PC-Max (fraction of test-visible 3D points not in training set)
  6. ConMax3D coverage (concept-pixel overlap)
  7. AlexNet entropy (JS divergence of softmax vs training mean)
  8. AlexNet embed dist (min cosine distance to training embeddings)
  9. DINOv2 embed dist (min cosine distance to training embeddings)
  10. CLIP embed dist (min cosine distance to training embeddings)
  11. InfoMax3D marginal (marginal log-det entropy gain)
  12. LPIPS dist (min LPIPS to training images)

Usage:
    python compute_perframe_distances.py --experiment v3_k15_random --scene fern
    python compute_perframe_distances.py --experiment v3_k15_random --scene fern --metrics geometric embedding
"""

import os
import sys
import json
import argparse
import glob
import struct
import collections
import math
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

WORKDIR = "/gpfs/workdir/malhotraa"
RESULTS_BASE = f"{WORKDIR}/ConMax3D_reproduce/results"
FEATURES_BASE = f"{RESULTS_BASE}/features"
OUTPUT_BASE = f"{RESULTS_BASE}/perframe"

LLFF_SCENES = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
TT_SCENES = ["Ballroom", "Barn", "Church", "Family", "Francis", "Horse", "Ignatius", "Museum"]
NS_SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

DATASET_CONFIGS = {
    "llff": {
        "scenes": LLFF_SCENES,
        "data_dir": f"{WORKDIR}/data/LLFF",
        "image_subdir": "images_4",
    },
    "tt": {
        "scenes": TT_SCENES,
        "data_dir": f"{WORKDIR}/data/Tanks",
        "image_subdir": "images",
    },
    "ns": {
        "scenes": NS_SCENES,
        "data_dir": f"{WORKDIR}/data/nerf_synthetic_gsplat",
        "image_subdir": "images",
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


# ============================================================================
# Pose loading utilities
# ============================================================================

def read_images_binary(path):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # Skip 2D points
            f.read(num_points2D * 24)  # x, y, point3D_id per point
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3D_binary(path):
    """Read COLMAP points3D.bin file."""
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            num_tracks = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(num_tracks):
                img_id = struct.unpack("<I", f.read(4))[0]
                pt2d_idx = struct.unpack("<I", f.read(4))[0]
                track.append((img_id, pt2d_idx))
            points[point_id] = {
                "xyz": np.array([x, y, z]),
                "rgb": np.array([r, g, b]),
                "error": error,
                "track": track,
            }
    return points


def qvec_to_rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    r = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # scipy uses xyzw
    return r.as_matrix()


def get_camera_center(qvec, tvec):
    """Get camera center in world coordinates: C = -R^T * t"""
    R = qvec_to_rotmat(qvec)
    return -R.T @ tvec


def get_cam2world(qvec, tvec):
    """Get 4x4 camera-to-world matrix."""
    R = qvec_to_rotmat(qvec)
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ tvec
    return c2w


def load_colmap_poses(scene_dir):
    """Load all camera poses from COLMAP sparse reconstruction."""
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    if not os.path.isdir(sparse_dir):
        return None, None

    images_bin = os.path.join(sparse_dir, "images.bin")
    if not os.path.exists(images_bin):
        return None, None

    images = read_images_binary(images_bin)

    # Sort by image name
    sorted_images = sorted(images.values(), key=lambda x: x["name"])

    cam2worlds = []
    centers = []
    names = []
    for img in sorted_images:
        c2w = get_cam2world(img["qvec"], img["tvec"])
        center = get_camera_center(img["qvec"], img["tvec"])
        cam2worlds.append(c2w)
        centers.append(center)
        names.append(img["name"])

    return {
        "cam2worlds": np.array(cam2worlds),
        "centers": np.array(centers),
        "names": names,
        "images_dict": images,
    }, sorted_images


def load_nerf_synthetic_poses(scene_dir):
    """Load poses from NeRF Synthetic transforms.json."""
    transforms_file = os.path.join(scene_dir, "transforms.json")
    if not os.path.exists(transforms_file):
        return None, None

    with open(transforms_file) as f:
        data = json.load(f)

    cam2worlds = []
    centers = []
    names = []
    for frame in data["frames"]:
        c2w = np.array(frame["transform_matrix"])
        cam2worlds.append(c2w)
        centers.append(c2w[:3, 3])
        names.append(frame.get("file_path", ""))

    # Also load test transforms
    test_file = os.path.join(scene_dir, "transforms_test.json")
    if os.path.exists(test_file):
        with open(test_file) as f:
            test_data = json.load(f)
        for frame in test_data["frames"]:
            c2w = np.array(frame["transform_matrix"])
            cam2worlds.append(c2w)
            centers.append(c2w[:3, 3])
            names.append(frame.get("file_path", ""))

    return {
        "cam2worlds": np.array(cam2worlds),
        "centers": np.array(centers),
        "names": names,
    }, None


# ============================================================================
# Geometric distance metrics
# ============================================================================

def fvs_baseline_distance(test_center, train_centers):
    """Min Euclidean camera center distance."""
    dists = np.linalg.norm(train_centers - test_center, axis=1)
    return float(np.min(dists))


def fvs_euclidean_distance(test_center, train_centers):
    """Min translation-only distance (same as baseline for COLMAP)."""
    return fvs_baseline_distance(test_center, train_centers)


def plucker_coords(c2w):
    """Compute 6D Plucker coordinates for a camera ray (viewing direction through center)."""
    center = c2w[:3, 3]
    direction = c2w[:3, 2]  # z-axis = viewing direction
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    moment = np.cross(center, direction)
    return np.concatenate([direction, moment])


def fvs_plucker_distance(test_c2w, train_c2ws):
    """Min 6D Plucker distance."""
    test_pl = plucker_coords(test_c2w)
    dists = []
    for tc2w in train_c2ws:
        train_pl = plucker_coords(tc2w)
        d = np.linalg.norm(test_pl - train_pl)
        dists.append(d)
    return float(np.min(dists))


def rotation_geodesic(R1, R2):
    """Geodesic distance between two rotation matrices."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    return float(angle)


def fvs_angular_distance(test_c2w, train_c2ws):
    """Min geodesic angular distance (rotation only)."""
    R_test = test_c2w[:3, :3]
    dists = []
    for tc2w in train_c2ws:
        R_train = tc2w[:3, :3]
        d = rotation_geodesic(R_test, R_train)
        dists.append(d)
    return float(np.min(dists))


# ============================================================================
# Point cloud coverage (PC-Max)
# ============================================================================

def compute_pc_max(test_image_id, train_image_ids, points3D):
    """Fraction of test-visible 3D points not visible in any training image."""
    # Collect 3D point IDs visible in test image
    test_point_ids = set()
    train_point_ids = set()

    for pid, pt in points3D.items():
        image_ids_in_track = {t[0] for t in pt["track"]}
        if test_image_id in image_ids_in_track:
            test_point_ids.add(pid)
        if image_ids_in_track & train_image_ids:
            train_point_ids.add(pid)

    if not test_point_ids:
        return 0.0

    uncovered = test_point_ids - train_point_ids
    return float(len(uncovered)) / float(len(test_point_ids))


# ============================================================================
# Embedding distance metrics
# ============================================================================

def min_cosine_distance(test_feat, train_feats):
    """Min cosine distance between test feature and training features."""
    # Normalize
    test_norm = test_feat / (torch.norm(test_feat) + 1e-8)
    train_norms = train_feats / (torch.norm(train_feats, dim=1, keepdim=True) + 1e-8)
    cos_sim = torch.mv(train_norms, test_norm)
    return float(1.0 - cos_sim.max().item())


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


# ============================================================================
# Main computation
# ============================================================================

def find_train_indices(experiment, scene):
    """Find train indices for an experiment/scene."""
    exp_dir = os.path.join(RESULTS_BASE, experiment, scene)
    pattern = os.path.join(exp_dir, f"train_indices_{scene}_*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[0]) as f:
            data = json.load(f)
        return data.get("selected_indices", [])
    return None


def get_image_paths_sorted(image_dir):
    """Get sorted list of image paths."""
    exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def compute_distances(experiment, scene, metric_groups=None, device="cuda"):
    """Compute all per-frame distance metrics for experiment/scene."""
    dataset_name = get_dataset_for_scene(scene)
    if dataset_name is None:
        print(f"Unknown scene: {scene}")
        return None

    cfg = DATASET_CONFIGS[dataset_name]
    scene_dir = os.path.join(cfg["data_dir"], scene)
    train_indices = find_train_indices(experiment, scene)
    if train_indices is None:
        print(f"No train indices for {experiment}/{scene}")
        return None

    if metric_groups is None:
        metric_groups = ["geometric", "embedding"]

    print(f"Processing {experiment}/{scene} (dataset={dataset_name})")
    print(f"  Train indices: {train_indices}")
    print(f"  Metric groups: {metric_groups}")

    # Load poses
    if dataset_name in ("llff", "tt"):
        pose_data, sorted_images = load_colmap_poses(scene_dir)
    else:
        pose_data, sorted_images = load_nerf_synthetic_poses(scene_dir)

    if pose_data is None:
        print(f"  Could not load poses for {scene}")
        return None

    n_total = len(pose_data["centers"])
    train_set = set(train_indices)
    test_indices = [i for i in range(n_total) if i not in train_set]
    print(f"  Total images: {n_total}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    train_centers = pose_data["centers"][train_indices]
    train_c2ws = pose_data["cam2worlds"][train_indices]

    results = []

    for ti in test_indices:
        frame_result = {"frame_id": ti}

        # ── Geometric metrics ──
        if "geometric" in metric_groups:
            test_center = pose_data["centers"][ti]
            test_c2w = pose_data["cam2worlds"][ti]

            frame_result["fvs_baseline"] = fvs_baseline_distance(test_center, train_centers)
            frame_result["fvs_euclidean"] = fvs_euclidean_distance(test_center, train_centers)
            frame_result["fvs_plucker"] = fvs_plucker_distance(test_c2w, train_c2ws)
            frame_result["fvs_angular"] = fvs_angular_distance(test_c2w, train_c2ws)

        # ── PC-Max (COLMAP only) ──
        if "pcmax" in metric_groups and dataset_name in ("llff", "tt"):
            points3d_path = os.path.join(scene_dir, "sparse", "0", "points3D.bin")
            if os.path.exists(points3d_path):
                points3D = read_points3D_binary(points3d_path)
                # Map sorted index to COLMAP image_id
                if sorted_images:
                    sorted_img_ids = [img["name"] for img in sorted(
                        pose_data.get("images_dict", {}).values(), key=lambda x: x["name"]
                    )]
                    # Need to map indices to COLMAP image IDs
                    img_id_map = {}
                    for img_id, img_data in pose_data.get("images_dict", {}).items() if hasattr(pose_data, "get") else []:
                        pass
                    # Simplified: use index-based mapping
                    all_img_ids = sorted(pose_data.get("images_dict", {}).keys())
                    if all_img_ids:
                        test_img_id = all_img_ids[ti] if ti < len(all_img_ids) else None
                        train_img_ids = {all_img_ids[i] for i in train_indices if i < len(all_img_ids)}
                        if test_img_id:
                            frame_result["pc_max"] = compute_pc_max(test_img_id, train_img_ids, points3D)

        results.append(frame_result)

    # ── Embedding distances ──
    if "embedding" in metric_groups:
        for model_name, feat_key in [("dinov2", "layer_8_cls"), ("alexnet", "fc6"), ("clip", "features")]:
            feat_path = os.path.join(FEATURES_BASE, dataset_name, model_name, f"{scene}.pt")
            if not os.path.exists(feat_path):
                print(f"  Features not found: {feat_path}")
                continue

            feat_data = torch.load(feat_path, map_location="cpu")

            if model_name == "dinov2":
                feats = feat_data.get("layer_8_cls")  # [N, D]
            elif model_name == "alexnet":
                feats = feat_data.get("fc6")  # [N, 4096]
            elif model_name == "clip":
                feats = feat_data.get("features")  # [N, 512]

            if feats is None:
                continue

            # Map indices to features (images may include test_images for NS)
            # For COLMAP datasets, features are extracted from all images
            n_feats = feats.shape[0]
            if n_feats < n_total:
                print(f"  Warning: {model_name} has {n_feats} features but {n_total} images. Skipping.")
                continue

            train_feats = feats[train_indices]

            metric_name = f"{model_name}_dist"
            for i, ti in enumerate(test_indices):
                if ti < n_feats:
                    test_feat = feats[ti]
                    results[i][metric_name] = min_cosine_distance(test_feat, train_feats)

            # AlexNet entropy (JS divergence of softmax)
            if model_name == "alexnet":
                softmax = feat_data.get("softmax")
                if softmax is not None:
                    train_softmax_mean = softmax[train_indices].numpy().mean(axis=0)
                    for i, ti in enumerate(test_indices):
                        if ti < softmax.shape[0]:
                            test_sm = softmax[ti].numpy()
                            results[i]["alexnet_entropy"] = js_divergence(test_sm, train_softmax_mean)

    # ── InfoMax3D marginal entropy ──
    if "infomax" in metric_groups:
        feat_path = os.path.join(FEATURES_BASE, dataset_name, "dinov2", f"{scene}.pt")
        if os.path.exists(feat_path):
            feat_data = torch.load(feat_path, map_location="cpu")
            cls_feats = feat_data.get("layer_8_cls")
            if cls_feats is not None and cls_feats.shape[0] >= n_total:
                train_feats_np = cls_feats[train_indices].numpy()
                # Compute covariance of training set
                cov_train = np.cov(train_feats_np.T) + 1e-6 * np.eye(train_feats_np.shape[1])
                sign, logdet_train = np.linalg.slogdet(cov_train)
                if sign > 0:
                    for i, ti in enumerate(test_indices):
                        test_feat_np = cls_feats[ti].numpy().reshape(1, -1)
                        augmented = np.vstack([train_feats_np, test_feat_np])
                        cov_aug = np.cov(augmented.T) + 1e-6 * np.eye(augmented.shape[1])
                        sign_aug, logdet_aug = np.linalg.slogdet(cov_aug)
                        if sign_aug > 0:
                            results[i]["infomax3d_marginal"] = float(logdet_aug - logdet_train)

    # ── LPIPS distance ──
    if "lpips" in metric_groups:
        try:
            import lpips as lpips_module
            from PIL import Image
            from torchvision import transforms

            lpips_fn = lpips_module.LPIPS(net="alex").to(device)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            image_dir = os.path.join(scene_dir, cfg["image_subdir"])
            if not os.path.isdir(image_dir):
                image_dir = os.path.join(scene_dir, "images")

            image_paths = get_image_paths_sorted(image_dir)

            if len(image_paths) >= n_total:
                # Preload and transform training images
                train_imgs = []
                for idx in train_indices:
                    img = Image.open(image_paths[idx]).convert("RGB")
                    img_t = transform(img).unsqueeze(0).to(device) * 2 - 1
                    train_imgs.append(img_t)
                train_batch = torch.cat(train_imgs, dim=0)

                with torch.no_grad():
                    for i, ti in enumerate(test_indices):
                        test_img = Image.open(image_paths[ti]).convert("RGB")
                        test_t = transform(test_img).unsqueeze(0).to(device) * 2 - 1
                        # Compute LPIPS to each training image
                        test_expanded = test_t.expand(len(train_indices), -1, -1, -1)
                        lpips_vals = lpips_fn(test_expanded, train_batch)
                        results[i]["lpips_dist"] = float(lpips_vals.min().item())
        except Exception as e:
            print(f"  LPIPS distance failed: {e}")

    # Save output
    output = {
        "experiment": experiment,
        "scene": scene,
        "dataset": dataset_name,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "n_test_frames": len(test_indices),
        "metric_groups": metric_groups,
        "per_frame": results,
    }

    out_dir = os.path.join(OUTPUT_BASE, experiment)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}_distances.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(results)} test frame distances to {out_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Compute per-frame distance metrics")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--scene", type=str, help="Scene name (if omitted, process all)")
    parser.add_argument("--metrics", nargs="+", default=["geometric", "embedding"],
                        choices=["geometric", "embedding", "pcmax", "infomax", "lpips"],
                        help="Metric groups to compute")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        exp_dir = os.path.join(RESULTS_BASE, args.experiment)
        scenes = sorted([
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
            and os.path.exists(os.path.join(exp_dir, d, f"train_indices_{d}_*.json".replace("*", "")))
        ])
        # Fallback: find train_indices files
        if not scenes:
            pattern = os.path.join(exp_dir, "*/train_indices_*.json")
            files = glob.glob(pattern)
            scenes = sorted(set(os.path.basename(os.path.dirname(f)) for f in files))
        print(f"Found {len(scenes)} scenes: {scenes}")

    for scene in scenes:
        try:
            compute_distances(args.experiment, scene, metric_groups=args.metrics, device=args.device)
        except Exception as e:
            print(f"ERROR {args.experiment}/{scene}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

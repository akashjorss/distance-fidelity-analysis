"""Unified camera pose loading (LLFF + COLMAP) and epipolar geometry utilities."""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Set


def load_poses_llff(basedir: str, factor: int = 4):
    """Load LLFF camera poses from poses_bounds.npy.

    Returns:
        c2w: (N, 3, 4) camera-to-world matrices
        camera_positions: (N, 3) camera centers in world frame
        focal: focal length (adjusted for factor)
        H, W: image dimensions (adjusted for factor)
    """
    poses_arr = np.load(Path(basedir) / "poses_bounds.npy")
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    # LLFF axis convention swap: [y, -x, z]
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
    )
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  # (N, 3, 5)

    H_orig, W_orig, focal_orig = poses[0, 0, -1], poses[0, 1, -1], poses[0, 2, -1]
    H = int(H_orig) // factor
    W = int(W_orig) // factor
    focal = float(focal_orig) / factor

    c2w = poses[:, :3, :4]  # (N, 3, 4)
    camera_positions = c2w[:, :3, 3].copy()  # (N, 3)
    return c2w, camera_positions, focal, H, W


def _quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_poses_colmap(basedir: str, factor: int = 1):
    """Load COLMAP poses from sparse/0/ text files.

    Returns same signature as load_poses_llff.
    """
    sparse_dir = Path(basedir) / "sparse" / "0"

    # Parse cameras.txt
    cameras = {}
    with open(sparse_dir / "cameras.txt", "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {"model": model, "width": width, "height": height, "params": params}

    # Parse images.txt (alternating: image line, points line)
    images = {}
    with open(sparse_dir / "images.txt", "r") as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    # Every pair of lines: (image_data, point_data)
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        R_w2c = _quat_to_rotmat(qw, qx, qy, qz)
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        c2w_34 = np.zeros((3, 4), dtype=np.float32)
        c2w_34[:3, :3] = R_c2w
        c2w_34[:3, 3] = t_c2w
        images[name] = {"c2w": c2w_34, "cam_id": cam_id, "position": t_c2w.astype(np.float32)}

    sorted_names = sorted(images.keys())
    c2w = np.stack([images[n]["c2w"] for n in sorted_names])
    camera_positions = np.stack([images[n]["position"] for n in sorted_names])

    first_cam = cameras[images[sorted_names[0]]["cam_id"]]
    focal = first_cam["params"][0] / factor
    H = first_cam["height"] // factor
    W = first_cam["width"] // factor
    return c2w, camera_positions, focal, H, W


def load_poses_auto(basedir: str, factor: int = 4):
    """Auto-detect LLFF vs COLMAP and load poses."""
    if (Path(basedir) / "poses_bounds.npy").exists():
        return load_poses_llff(basedir, factor=factor)
    elif (Path(basedir) / "sparse" / "0" / "images.txt").exists():
        return load_poses_colmap(basedir, factor=factor)
    else:
        raise FileNotFoundError(f"No poses_bounds.npy or sparse/0/ in {basedir}")


def compute_intrinsic_matrix(focal: float, H: int, W: int) -> np.ndarray:
    """Construct 3x3 intrinsic matrix K with principal point at image center."""
    return np.array(
        [[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]], dtype=np.float64
    )


def compute_fundamental_matrix(c2w_i, c2w_j, K):
    """Compute fundamental matrix F such that x_j^T F x_i = 0.

    Args:
        c2w_i, c2w_j: (3,4) camera-to-world matrices
        K: (3,3) intrinsic matrix
    Returns:
        F: (3,3) fundamental matrix
    """
    c2w_i = c2w_i.astype(np.float64)
    c2w_j = c2w_j.astype(np.float64)

    R_i, t_i = c2w_i[:3, :3], c2w_i[:3, 3]
    R_j, t_j = c2w_j[:3, :3], c2w_j[:3, 3]

    R_w2c_j = R_j.T
    # Relative pose: camera j coordinate frame relative to camera i
    R_rel = R_w2c_j @ R_i
    t_rel = R_w2c_j @ (t_i - t_j)

    # Essential matrix E = [t_rel]_x @ R_rel
    tx = np.array(
        [[0, -t_rel[2], t_rel[1]], [t_rel[2], 0, -t_rel[0]], [-t_rel[1], t_rel[0], 0]],
        dtype=np.float64,
    )
    E = tx @ R_rel

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    norm = np.linalg.norm(F)
    if norm > 1e-12:
        F = F / norm
    return F


def symmetric_epipolar_distance(
    centroid_A: np.ndarray,
    centroid_B: np.ndarray,
    F_AB: np.ndarray,
) -> float:
    """Compute symmetric epipolar distance between two points.

    Given F such that x_B^T F x_A = 0 for corresponding points,
    computes the average of:
      - distance from centroid_B to epipolar line of centroid_A in image B
      - distance from centroid_A to epipolar line of centroid_B in image A

    Args:
        centroid_A: (2,) as (row, col) in image A
        centroid_B: (2,) as (row, col) in image B
        F_AB: (3,3) fundamental matrix (x_B^T F x_A = 0)

    Returns:
        Symmetric epipolar distance in pixels.
    """
    # Homogeneous coords: (col, row, 1) = (x, y, 1)
    x_A = np.array([centroid_A[1], centroid_A[0], 1.0], dtype=np.float64)
    x_B = np.array([centroid_B[1], centroid_B[0], 1.0], dtype=np.float64)

    # Epipolar line of A in image B: l_B = F @ x_A
    l_B = F_AB @ x_A
    denom_B = np.sqrt(l_B[0] ** 2 + l_B[1] ** 2)
    d_B = abs(x_B @ l_B) / denom_B if denom_B > 1e-12 else 1e6

    # Epipolar line of B in image A: l_A = F^T @ x_B
    l_A = F_AB.T @ x_B
    denom_A = np.sqrt(l_A[0] ** 2 + l_A[1] ** 2)
    d_A = abs(x_A @ l_A) / denom_A if denom_A > 1e-12 else 1e6

    return (d_A + d_B) / 2.0

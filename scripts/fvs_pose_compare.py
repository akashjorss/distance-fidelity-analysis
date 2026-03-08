"""Compare FVS selections: Euclidean, Angular, Plucker L2 on camera poses."""
import sys, os, struct, collections, argparse
import numpy as np
from pathlib import Path

# ── Read COLMAP images.bin ──
CameraImage = collections.namedtuple("CameraImage", ["id", "qvec", "tvec", "camera_id", "name"])

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)  # skip 2D points
            images[img_id] = CameraImage(img_id, qvec, tvec, camera_id, name)
    return images

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def get_camera_centers_and_dirs(images):
    """Extract camera center (C = -R^T t) and viewing direction (3rd row of R)."""
    sorted_imgs = sorted(images.values(), key=lambda x: x.name)
    centers = []
    directions = []
    names = []
    for img in sorted_imgs:
        R = qvec2rotmat(img.qvec)
        C = -R.T @ img.tvec
        d = R[2, :]  # viewing direction (z-axis of camera)
        centers.append(C)
        directions.append(d / np.linalg.norm(d))
        names.append(img.name)
    return np.array(centers), np.array(directions), names

def fvs_greedy(dist_matrix, k, seed=0):
    """Greedy max-min distance selection."""
    N = dist_matrix.shape[0]
    seed = seed % N
    selected = [seed]
    remaining = set(range(N)) - {seed}
    for _ in range(k - 1):
        if not remaining:
            break
        candidates = list(remaining)
        sel_dists = dist_matrix[np.ix_(candidates, selected)]
        min_dists = sel_dists.min(axis=1)
        best = candidates[np.argmax(min_dists)]
        selected.append(best)
        remaining.discard(best)
    return sorted(selected)

def plucker_coordinates(center, direction):
    """Plucker representation: (d, m) where m = c x d."""
    m = np.cross(center, direction)
    return np.concatenate([direction, m])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()

    images_bin = os.path.join(args.data_dir, args.scene, "sparse", "0", "images.bin")
    if not os.path.exists(images_bin):
        print(f"ERROR: {images_bin} not found")
        sys.exit(1)

    images = read_images_binary(images_bin)
    centers, directions, names = get_camera_centers_and_dirs(images)
    N = len(centers)

    if args.max_images > 0 and N > args.max_images:
        centers = centers[:args.max_images]
        directions = directions[:args.max_images]
        names = names[:args.max_images]
        N = args.max_images

    print(f"Scene: {args.scene}, N={N} images, k={args.k}")

    # 1. Euclidean distance on camera centers
    diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 3)
    dist_euclidean = np.linalg.norm(diff, axis=2)

    # 2. Angular distance (geodesic on unit sphere of viewing directions)
    cos_sim = np.clip(directions @ directions.T, -1, 1)
    dist_angular = np.arccos(cos_sim)  # radians

    # 3. Plucker L2 distance
    pluckers = np.array([plucker_coordinates(c, d) for c, d in zip(centers, directions)])  # (N, 6)
    diff_p = pluckers[:, None, :] - pluckers[None, :, :]
    dist_plucker = np.linalg.norm(diff_p, axis=2)

    sel_eucl = fvs_greedy(dist_euclidean, args.k, seed=42)
    sel_angu = fvs_greedy(dist_angular, args.k, seed=42)
    sel_plck = fvs_greedy(dist_plucker, args.k, seed=42)

    print(f"  Euclidean:  {sel_eucl}")
    print(f"  Angular:    {sel_angu}")
    print(f"  Plucker L2: {sel_plck}")

    # Overlap
    s_e, s_a, s_p = set(sel_eucl), set(sel_angu), set(sel_plck)
    print(f"  Eucl∩Angu: {len(s_e & s_a)}/{args.k}")
    print(f"  Eucl∩Plck: {len(s_e & s_p)}/{args.k}")
    print(f"  Angu∩Plck: {len(s_a & s_p)}/{args.k}")
    print(f"  All three: {len(s_e & s_a & s_p)}/{args.k}")

if __name__ == "__main__":
    main()

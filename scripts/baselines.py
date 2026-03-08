"""Baseline frame selection methods for ConMax3D reproduction."""

import numpy as np
import json
import os
import argparse

def load_poses(basedir, factor=4):
    """Load camera poses from LLFF poses_bounds.npy."""
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    # Correct rotation matrix ordering
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    return poses[:, :3, :4]  # (N, 3, 4)

def random_select(n_images, k, seed=42):
    """Randomly select k frames from n_images."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n_images, k, replace=False).tolist())

def fvs_select(poses, k, seed=42):
    """Furthest View Sampling: iteratively select the frame furthest from the current set.
    
    Args:
        poses: (N, 3, 4) camera poses
        k: number of frames to select
        seed: random seed for first frame selection
    
    Returns:
        List of selected frame indices
    """
    n = len(poses)
    camera_positions = poses[:, :3, 3]  # (N, 3)
    
    # Start with a random frame
    rng = np.random.RandomState(seed)
    selected = [rng.randint(n)]
    remaining = set(range(n)) - set(selected)
    
    while len(selected) < k and remaining:
        # For each remaining frame, compute min distance to any selected frame
        selected_positions = camera_positions[selected]
        best_idx = None
        best_min_dist = -1
        
        for idx in remaining:
            dists = np.linalg.norm(camera_positions[idx] - selected_positions, axis=1)
            min_dist = dists.min()
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return selected

def store_train_indices(train_indices, output_file, scene):
    """Store train indices in JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[scene] = [int(x) for x in train_indices]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def number_to_words(n):
    return {10: 'ten', 20: 'twenty', 25: 'twenty_five'}.get(n, str(n))


def load_poses_nerf_synthetic(basedir):
    """Load camera poses from NeRF Synthetic transforms_train.json."""
    transforms_path = os.path.join(basedir, 'transforms_train.json')
    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    poses = []
    for frame in meta['frames']:
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        if transform_matrix.shape == (3, 4):
            full = np.eye(4, dtype=np.float32)
            full[:3, :] = transform_matrix
            transform_matrix = full
        poses.append(transform_matrix[:3, :4])

    poses = np.stack(poses, axis=0)  # (N, 3, 4)
    return poses


def count_images_nerf_synthetic(basedir):
    """Count training images in NeRF Synthetic train/ directory."""
    train_dir = os.path.join(basedir, 'train')
    if os.path.isdir(train_dir):
        return len([f for f in os.listdir(train_dir) if f.lower().endswith('.png')])
    # Fallback: count from transforms_train.json
    transforms_path = os.path.join(basedir, 'transforms_train.json')
    with open(transforms_path, 'r') as f:
        meta = json.load(f)
    return len(meta['frames'])

def main():
    parser = argparse.ArgumentParser(description='Generate baseline frame selection indices')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to LLFF scene directory')
    parser.add_argument('--scene', type=str, required=True, help='Scene name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--method', type=str, choices=['random', 'fvs'], required=True)
    parser.add_argument('--dataset_type', type=str, default='llff',
                        choices=['llff', 'nerf_synthetic'],
                        help='Dataset type for pose loading')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Count images
    if args.dataset_type == 'nerf_synthetic':
        n_images = count_images_nerf_synthetic(args.data_dir)
    else:
        img_dir = os.path.join(args.data_dir, 'images')
        n_images = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f'Scene {args.scene}: {n_images} images')
    
    frame_counts = [10, 20, 25]
    max_k = min(max(frame_counts), n_images)
    
    if args.method == 'random':
        indices = random_select(n_images, max_k, seed=args.seed)
    elif args.method == 'fvs':
        if args.dataset_type == 'nerf_synthetic':
            poses = load_poses_nerf_synthetic(args.data_dir)
        else:
            poses = load_poses(args.data_dir)
        indices = fvs_select(poses, max_k, seed=args.seed)
    
    for count in frame_counts:
        if count > n_images:
            print(f'Skipping k={count} (only {n_images} images)')
            continue
        count_str = number_to_words(count)
        output_file = os.path.join(args.output_dir, count_str, args.method, 'train_indices.json')
        store_train_indices(indices[:count], output_file, args.scene)
        print(f'Stored {count} {args.method} indices for {args.scene}')

if __name__ == '__main__':
    main()

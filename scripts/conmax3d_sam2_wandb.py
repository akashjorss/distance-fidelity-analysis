"""ConMax3D frame selection with SAM2 + W&B logging and configurable hyperparameters."""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import random
import networkx as nx
from collections import Counter
import argparse
import json
import time
from functools import wraps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

# Add parent directory to path for load_llff
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Timing
time_taken = {}
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        time_taken[func.__name__] = elapsed
        print(f"  [{func.__name__}] took {elapsed:.1f}s")
        return result
    return wrapper

# ========== Data Loading ==========
def load_data(data_path, factor=4):
    from load_llff import load_llff_data
    imgs, poses, bds, render_poses, i_test = load_llff_data(data_path, factor=factor)
    hwf = poses[0, :3, -1]
    H, W, focal = hwf[0], hwf[1], hwf[2]
    poses = poses[:, :3, :4]
    images = (imgs * 255).astype(np.uint8)
    return images, poses, H, W


def load_data_nerf_synthetic(data_path, factor=1):
    """Load images from NeRF Synthetic train/ directory with RGBA -> RGB compositing."""
    import glob
    train_dir = os.path.join(data_path, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train/ directory not found at {train_dir}")

    # Sort image files numerically
    img_files = sorted(glob.glob(os.path.join(train_dir, "*.png")),
                       key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

    if not img_files:
        raise FileNotFoundError(f"No PNG files found in {train_dir}")

    images = []
    for img_file in img_files:
        img = Image.open(img_file).convert("RGBA")
        img_np = np.array(img).astype(np.float32) / 255.0
        # Alpha composite on white background
        alpha = img_np[:, :, 3:4]
        rgb = img_np[:, :, :3] * alpha + (1.0 - alpha)
        images.append((rgb * 255).astype(np.uint8))

    images = np.stack(images, axis=0)
    H, W = images.shape[1], images.shape[2]

    # No poses needed for ConMax3D (only needs images)
    poses = None
    print(f"Loaded {len(images)} NeRF Synthetic images ({H}x{W})")
    return images, poses, H, W

# ========== Mask Generation ==========
@timeit
def generate_sam2_masks(images, mask_generator):
    masks = []
    for image in tqdm(images, desc="Generating SAM2 masks"):
        mask = mask_generator.generate(image)
        masks.append(mask)
    return masks

def delete_small_masks(masks, H, W, min_num_pixels=None):
    min_pixels = np.sqrt(H * W) if min_num_pixels is None else min_num_pixels
    filtered_masks = []
    for mask_set in masks:
        filtered_masks.append([m for m in mask_set if np.sum(m['segmentation']) > min_pixels])
    return filtered_masks

# ========== Image Processing ==========
@timeit
def crop_images_with_masks(images, masks):
    cropped_images = []
    cropped_images_to_images = {}
    for i in tqdm(range(len(images)), desc="Cropping images"):
        for j, mask in enumerate(masks[i]):
            mask_3d = np.dstack([mask['segmentation']] * 3)
            cropped_img = Image.fromarray(images[i] * mask_3d)
            cropped_images.append(cropped_img)
            cropped_images_to_images[len(cropped_images) - 1] = i
    return cropped_images, cropped_images_to_images

@timeit
def generate_image_embeddings(cropped_images, img2vec, batch_size=16):
    img_vectors = []
    for i in tqdm(range(0, len(cropped_images), batch_size), desc="Generating embeddings"):
        batch = cropped_images[i:i + batch_size]
        batch_vectors = img2vec.get_vec(batch)
        img_vectors.extend(batch_vectors)
    return np.array(img_vectors)

# ========== Clustering ==========
@timeit
def perform_clustering(distance_matrix, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='precomputed',
        cluster_selection_method='eom',
    )
    return clusterer.fit_predict(distance_matrix)

# ========== Graph & Selection ==========
def cropped_image_to_pixel_ids(cropped_image):
    arr = np.array(cropped_image)[:, :, 0]
    i, j = np.where(arr > 0)
    return (i * arr.shape[1] + j).tolist()

def calculate_pixel_contribution(current_selection, candidate_image, G):
    new_pixels = set()
    for concept in G.neighbors(candidate_image):
        if G.nodes[concept]['type'] == 'concept':
            current_pixels = set()
            for img in current_selection:
                for bag_node in G.neighbors(img):
                    if G.nodes[bag_node]['type'] == 'pixel_bag' and bag_node in G.neighbors(concept):
                        current_pixels.update(G.nodes[bag_node]['value'])
            candidate_pixels = set()
            for bag_node in G.neighbors(candidate_image):
                if G.nodes[bag_node]['type'] == 'pixel_bag' and bag_node in G.neighbors(concept):
                    candidate_pixels.update(G.nodes[bag_node]['value'])
            new_pixels.update(candidate_pixels - current_pixels)
    return len(new_pixels)

@timeit
def greedy_select_images(G, k):
    all_images = [n for n in G.nodes if G.nodes[n]['type'] == 'image']
    selected = []
    remaining = set(all_images)
    while len(selected) < k and remaining:
        best_image = None
        max_contribution = 0
        for image in tqdm(remaining, desc=f"Selecting frame {len(selected)+1}/{k}"):
            c = calculate_pixel_contribution(selected, image, G)
            if c > max_contribution:
                max_contribution = c
                best_image = image
        if best_image is not None:
            selected.append(best_image)
            remaining.remove(best_image)
        else:
            rand_idx = random.choice(list(remaining))
            selected.append(rand_idx)
            remaining.remove(rand_idx)
    return selected

@timeit
def construct_graph(num_images, cropped_images, cluster_labels, cropped_images_to_images):
    G = nx.Graph()
    for i in range(num_images):
        G.add_node(i, type="image")
    for i in range(len(cropped_images)):
        G.add_node(f"{cropped_images_to_images[i]}.{i}", type="mask")
    for i in range(len(cluster_labels)):
        if cluster_labels[i] != -1:
            G.add_node(f"concept_{cluster_labels[i]}", type="concept")
    for i in range(len(cropped_images)):
        G.add_edge(f"{cropped_images_to_images[i]}.{i}", cropped_images_to_images[i], type="has_mask")
    for i in range(len(cropped_images)):
        if cluster_labels[i] != -1:
            G.add_edge(f"concept_{cluster_labels[i]}", f"{cropped_images_to_images[i]}.{i}", type="has_concept")
            G.add_edge(cropped_images_to_images[i], f"concept_{cluster_labels[i]}", type="has_concept")
    for i in tqdm(range(len(cropped_images)), desc="Adding pixel bags"):
        if cluster_labels[i] == -1:
            continue
        pixel_ids = cropped_image_to_pixel_ids(cropped_images[i])
        G.add_node(f"pixel_bag_{i}", type="pixel_bag", value=pixel_ids)
        G.add_edge(f"{cropped_images_to_images[i]}.{i}", f"pixel_bag_{i}", type="has_pixel_bag")
        G.add_edge(cropped_images_to_images[i], f"pixel_bag_{i}", type="has_pixel_bag")
        G.add_edge(f"concept_{cluster_labels[i]}", f"pixel_bag_{i}", type="has_pixel_bag")
    return G

# ========== Main Pipeline ==========
def select_frames(base_dir, scene, num_frames, config):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from img2vec_pytorch import Img2Vec

    print(f"Processing scene: {scene}")
    data_path = os.path.join(base_dir, scene)
    # Load data based on dataset type
    if config.get('dataset_type', 'llff') == 'nerf_synthetic':
        images, poses, H, W = load_data_nerf_synthetic(data_path, factor=config.get('downscale_factor', 1))
    else:
        images, poses, H, W = load_data(data_path, factor=config['downscale_factor'])

    # SAM2 Model
    sam2_checkpoint = config['sam2_checkpoint']
    model_cfg = config['sam2_model_cfg']
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        pred_iou_thresh=config['pred_iou_thresh']
    )

    # Generate masks
    masks = generate_sam2_masks(images, mask_generator)
    masks = delete_small_masks(masks, H, W, min_num_pixels=config.get('min_mask_pixels'))
    total_masks = sum(len(m) for m in masks)
    print(f"Total masks after filtering: {total_masks}")

    # Crop images
    cropped_images, cropped_images_to_images = crop_images_with_masks(images, masks)
    print(f"Number of cropped images: {len(cropped_images)}")

    # Embeddings
    img2vec = Img2Vec(cuda=torch.cuda.is_available(), model=config['efficientnet_model'])
    img_vectors = generate_image_embeddings(cropped_images, img2vec, batch_size=config.get('embedding_batch_size', 16))

    # Clustering
    distance_matrix = cosine_distances(img_vectors).astype(np.float64)
    min_cluster_size = max(2, int(len(images) * config['min_cluster_factor']))
    cluster_labels = perform_clustering(distance_matrix, min_cluster_size)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = Counter(cluster_labels)[-1] if -1 in cluster_labels else 0
    print(f"Clusters: {n_clusters}, Outliers: {n_outliers}/{len(cluster_labels)}")

    # Graph + Selection
    G = construct_graph(len(images), cropped_images, cluster_labels, cropped_images_to_images)
    selected_images = greedy_select_images(G, num_frames)
    print(f"Selected images: {selected_images}")

    # Return metadata for logging
    metadata = {
        'n_images': len(images),
        'total_masks': total_masks,
        'n_cropped': len(cropped_images),
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_ratio': n_outliers / len(cluster_labels) if len(cluster_labels) > 0 else 0,
        'time_taken': dict(time_taken),
    }
    return selected_images, metadata

def number_to_words(n):
    return {10: 'ten', 20: 'twenty', 25: 'twenty_five'}.get(n, str(n))

def store_train_indices(train_indices, output_dir, scene):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "train_indices.json")
    data = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    data[scene] = [int(x) for x in train_indices]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='ConMax3D frame selection with W&B')
    parser.add_argument('--base_dir', type=str, required=True, help='Base dir with scene subdirs')
    parser.add_argument('--scene', type=str, required=True, help='Scene name')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir for train_indices')
    parser.add_argument('--num_frames', type=int, default=25, help='Max frames to select')
    # Hyperparameters
    parser.add_argument('--pred_iou_thresh', type=float, default=0.8)
    parser.add_argument('--min_cluster_factor', type=float, default=0.25, help='min_cluster_size = N * factor')
    parser.add_argument('--efficientnet_model', type=str, default='efficientnet_b0')
    parser.add_argument('--downscale_factor', type=int, default=4)
    parser.add_argument('--dataset_type', type=str, default='llff',
                        choices=['llff', 'nerf_synthetic'],
                        help='Dataset type: llff or nerf_synthetic')
    parser.add_argument('--min_mask_pixels', type=float, default=None, help='Min pixels per mask (default: sqrt(H*W))')
    parser.add_argument('--embedding_batch_size', type=int, default=16)
    # SAM2 config
    parser.add_argument('--sam2_checkpoint', type=str, 
                        default='/gpfs/workdir/malhotraa/segment-anything-2/checkpoints/sam2_hiera_large.pt')
    parser.add_argument('--sam2_model_cfg', type=str, default='sam2_hiera_l.yaml')
    # W&B
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='conmax3d-reproduction')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()

    config = {
        'pred_iou_thresh': args.pred_iou_thresh,
        'min_cluster_factor': args.min_cluster_factor,
        'efficientnet_model': args.efficientnet_model,
        'downscale_factor': args.downscale_factor,
        'min_mask_pixels': args.min_mask_pixels,
        'embedding_batch_size': args.embedding_batch_size,
        'dataset_type': args.dataset_type,
        'sam2_checkpoint': args.sam2_checkpoint,
        'sam2_model_cfg': args.sam2_model_cfg,
    }

    # W&B init
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"conmax3d_{args.scene}",
            config=config,
        )
        config = dict(wandb.config)  # Use W&B sweep config if available
        # Re-apply SAM2 paths (not swept)
        config['sam2_checkpoint'] = args.sam2_checkpoint
        config['sam2_model_cfg'] = args.sam2_model_cfg

    # Run
    selected_images, metadata = select_frames(args.base_dir, args.scene, args.num_frames, config)

    # Store results
    frame_counts = [10, 20, 25]
    for count in frame_counts:
        if count > len(selected_images):
            continue
        count_str = number_to_words(count)
        output_path = os.path.join(args.output_dir, count_str, 'conmax3d')
        store_train_indices(selected_images[:count], output_path, args.scene)

    # Store timing
    time_dir = os.path.join(args.output_dir, 'time')
    os.makedirs(time_dir, exist_ok=True)
    with open(os.path.join(time_dir, f'execution_time_{args.scene}.json'), 'w') as f:
        json.dump(time_taken, f, indent=4)

    # Log to W&B
    if args.use_wandb:
        import wandb
        wandb.log({
            'n_clusters': metadata['n_clusters'],
            'n_outliers': metadata['n_outliers'],
            'outlier_ratio': metadata['outlier_ratio'],
            'total_masks': metadata['total_masks'],
            'n_cropped': metadata['n_cropped'],
            'selected_indices': selected_images,
            **{f'time/{k}': v for k, v in metadata['time_taken'].items()},
        })
        wandb.finish()

    print("Done!")

if __name__ == "__main__":
    main()

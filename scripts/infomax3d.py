"""
InfoMax3D: Maximum-Entropy Frame Selection for 3D Reconstruction.

Pose-free frame selection that maximizes per-cell Shannon entropy of dense
feature maps via log-determinant of Gram matrices (DPP formulation).

Pipeline:
  1. Extract dense features (DINOv2 or ResNet) → (N, H', W', C) tensors
     Supports multi-layer extraction (LPIPS-style) for combined entropy
  2. Seed: pick frame with highest internal feature diversity
  3. Greedy: select frame maximizing Σ_{layers} Σ_{r,c} log-det gain
  4. FVS baseline: cosine-distance furthest-view selection on global features

Usage:
  python infomax3d.py \
    --data_dir /path/to/dataset --scene fern \
    --output_dir /path/to/results --k 10
  # Multi-layer:
  python infomax3d.py ... --backbone dinov2 --dino_layers 4,6,8
  python infomax3d.py ... --backbone resnet --resnet_stages 2,3,4
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from functools import wraps
from typing import List, Tuple, Optional

# ─── Device setup ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
print(f"Device: {device}")

# ─── Timing ───
time_taken = {}


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        time_taken[func.__name__] = time.time() - t0
        print(f"  [{func.__name__}] {time_taken[func.__name__]:.1f}s")
        return result
    return wrapper


# ═══════════════════════════════════════════════════════
# Image Loading
# ═══════════════════════════════════════════════════════

def load_image_paths(scene_dir: str, factor: int, max_images: Optional[int] = None) -> List[str]:
    """Load sorted image paths from a COLMAP/LLFF scene directory."""
    img_dir = os.path.join(scene_dir, f"images_{factor}")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_dir, "images")

    fnames = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    paths = [os.path.join(img_dir, f) for f in fnames]

    if max_images and len(paths) > max_images:
        stride = len(paths) / max_images
        subset = [int(i * stride) for i in range(max_images)]
        paths = [paths[i] for i in subset]
        print(f"  Subsampled to {len(paths)} images (stride ~{stride:.1f})")

    print(f"  Found {len(paths)} images in {img_dir}")
    return paths


# ═══════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════

@timeit
def extract_features_dinov2(
    image_paths: List[str],
    layer: int = 8,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract dense features from DINOv2 ViT-B/14 at a given intermediate layer.

    Returns:
        features: (N, H', W', C) tensor of L2-normalized patch features on CPU
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.to(device).eval()

    # Hook to capture intermediate layer output
    intermediate_output = {}

    def hook_fn(module, input, output):
        intermediate_output["feat"] = output

    handle = model.blocks[layer].register_forward_hook(hook_fn)

    # DINOv2 ViT-B/14 expects 518x518 (37x37 patches of 14px)
    # but also works with 224x224 (16x16 patches)
    # Use 518 for higher spatial resolution
    img_size = 518
    patch_size = 14
    grid_size = img_size // patch_size  # 37

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            # Forward pass through model — we only need the hook output
            _ = model(batch)
            feat = intermediate_output["feat"]  # (B, 1+H'*W', C) with CLS token
            feat = feat[:, 1:, :]  # remove CLS token → (B, H'*W', C)
            B, HW, C = feat.shape
            feat = feat.reshape(B, grid_size, grid_size, C)
            # L2 normalize
            feat = F.normalize(feat.float(), dim=-1)
            all_features.append(feat.cpu())

        if (i // batch_size) % 5 == 0:
            print(f"    Batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")

    handle.remove()
    features = torch.cat(all_features, dim=0)  # (N, H', W', C)
    print(f"  Feature shape: {features.shape}")
    return features


@timeit
def extract_features_resnet(
    image_paths: List[str],
    stage: int = 3,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract dense features from ResNet-50 at a given stage (1-4).

    Stage feature map sizes (for 224x224 input):
      stage 1: 56x56x256, stage 2: 28x28x512, stage 3: 14x14x1024, stage 4: 7x7x2048

    Returns:
        features: (N, H', W', C) tensor of L2-normalized features on CPU
    """
    import torchvision.models as models

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet = resnet.to(device).eval()

    # Build a truncated model up to the desired stage
    layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]
    stage_blocks = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    for s in range(stage):
        layers.append(stage_blocks[s])
    feature_extractor = torch.nn.Sequential(*layers).to(device).eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            feat = feature_extractor(batch)  # (B, C, H', W')
            feat = feat.permute(0, 2, 3, 1)  # (B, H', W', C)
            feat = F.normalize(feat.float(), dim=-1)
            all_features.append(feat.cpu())

    features = torch.cat(all_features, dim=0)
    print(f"  Feature shape: {features.shape}")
    return features


# ═══════════════════════════════════════════════════════
# Multi-Layer Feature Extraction (LPIPS-style)
# ═══════════════════════════════════════════════════════

@timeit
def extract_features_dinov2_multilayer(
    image_paths: List[str],
    layers: List[int],
    batch_size: int = 8,
) -> List[torch.Tensor]:
    """Extract dense features from multiple DINOv2 layers in one forward pass.

    Returns:
        List of (N, H', W', C) tensors, one per layer
    """
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.to(device).eval()

    # Hook multiple layers
    intermediate_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            intermediate_outputs[layer_idx] = output
        return hook_fn

    handles = []
    for layer_idx in layers:
        h = model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    img_size = 518
    patch_size = 14
    grid_size = img_size // patch_size  # 37

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Accumulate per-layer features
    all_layer_features = {l: [] for l in layers}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            _ = model(batch)
            for layer_idx in layers:
                feat = intermediate_outputs[layer_idx]
                feat = feat[:, 1:, :]  # remove CLS token
                B, HW, C = feat.shape
                feat = feat.reshape(B, grid_size, grid_size, C)
                feat = F.normalize(feat.float(), dim=-1)
                all_layer_features[layer_idx].append(feat.cpu())

    for h in handles:
        h.remove()

    result = []
    for layer_idx in layers:
        features = torch.cat(all_layer_features[layer_idx], dim=0)
        print(f"  Layer {layer_idx} feature shape: {features.shape}")
        result.append(features)

    return result


@timeit
def extract_features_resnet_multilayer(
    image_paths: List[str],
    stages: List[int],
    batch_size: int = 8,
) -> List[torch.Tensor]:
    """Extract dense features from multiple ResNet stages in one forward pass.

    Returns:
        List of (N, H', W', C) tensors, one per stage
    """
    import torchvision.models as models

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet = resnet.to(device).eval()

    # Build feature extractors for each stage
    stem = torch.nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    stage_blocks = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    max_stage = max(stages)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_stage_features = {s: [] for s in stages}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            x = stem(batch)
            for s_idx in range(max_stage):
                x = stage_blocks[s_idx](x)
                stage_num = s_idx + 1
                if stage_num in stages:
                    feat = x.permute(0, 2, 3, 1)  # (B, H', W', C)
                    feat = F.normalize(feat.float(), dim=-1)
                    all_stage_features[stage_num].append(feat.cpu())

    result = []
    for stage_num in stages:
        features = torch.cat(all_stage_features[stage_num], dim=0)
        print(f"  Stage {stage_num} feature shape: {features.shape}")
        result.append(features)

    return result


# ═══════════════════════════════════════════════════════
# InfoMax Selection (Log-Det Entropy Maximization)
# ═══════════════════════════════════════════════════════

def seed_frame(features: torch.Tensor) -> int:
    """Pick the frame with highest internal feature diversity.

    Diversity = mean std of features across spatial cells.
    """
    # features: (N, H', W', C)
    N = features.shape[0]
    diversities = []
    for i in range(N):
        # std across spatial dim for each channel, then mean
        std_per_channel = features[i].std(dim=(0, 1))  # (C,)
        diversities.append(std_per_channel.mean().item())
    best = int(np.argmax(diversities))
    print(f"  Seed frame: {best} (diversity={diversities[best]:.4f})")
    return best


@timeit
def greedy_entropy_select(
    features: torch.Tensor,
    k: int,
    eps: float = 1e-6,
) -> List[int]:
    """Greedy frame selection maximizing per-cell log-det entropy.

    Uses Schur complement for efficient marginal gain computation.
    Maintains per-cell Cholesky factors for numerical stability.

    Args:
        features: (N, H, W, C) L2-normalized feature tensors
        k: number of frames to select
        eps: nugget for numerical stability

    Returns:
        List of selected frame indices
    """
    N, H, W, C = features.shape
    n_cells = H * W

    # Reshape to (N, n_cells, C) for vectorized operations
    feats = features.reshape(N, n_cells, C).to(device)  # (N, n_cells, C)

    # Step 1: Seed
    seed = seed_frame(features)
    selected = [seed]
    remaining = set(range(N)) - {seed}

    # Step 2: Initialize per-cell Cholesky factor L
    # For the seed, Gram matrix is 1x1: [f·f + eps] = [1 + eps] (since L2 normalized)
    # L is lower triangular: L[cell] is (m, m) where m = len(selected)
    # We'll maintain L_inv for fast Schur complement: L_inv @ k_vec

    # Per-cell: L (Cholesky of Gram matrix of selected features at that cell)
    # Start with seed: G = [[1+eps]], L = [[sqrt(1+eps)]]
    seed_feats = feats[seed]  # (n_cells, C)
    k_self_seed = (seed_feats * seed_feats).sum(dim=-1) + eps  # (n_cells,)
    L_diag = k_self_seed.sqrt()  # (n_cells,) — 1x1 Cholesky

    # Store selected features per cell for Gram operations
    # selected_feats[cell] is (m, C) stack of selected features at cell
    # We'll use a tensor: (n_cells, max_k, C)
    sel_feats = torch.zeros(n_cells, k, C, device=device, dtype=feats.dtype)
    sel_feats[:, 0, :] = seed_feats

    # Store L as dense lower-triangular: (n_cells, k, k)
    L = torch.zeros(n_cells, k, k, device=device, dtype=torch.float32)
    L[:, 0, 0] = L_diag.float()

    m = 1  # number currently selected

    for step in range(1, k):
        candidates = list(remaining)
        if not candidates:
            break

        cand_feats = feats[candidates]  # (n_cand, n_cells, C)
        n_cand = len(candidates)

        # For each candidate, compute per-cell Schur complement gain:
        # gain_cell = k_self - k_vec^T G^{-1} k_vec
        #           = k_self - ||L^{-1} k_vec||^2
        # where k_self = f_c · f_c + eps
        #       k_vec[j] = f_c · f_selected_j  (for j in selected, at this cell)

        # k_self: (n_cand, n_cells)
        k_self = (cand_feats * cand_feats).sum(dim=-1).float() + eps

        # k_vec: (n_cand, n_cells, m) — dot products with selected features
        # sel_feats[:, :m, :] is (n_cells, m, C)
        # cand_feats is (n_cand, n_cells, C)
        # k_vec[c, cell, j] = cand_feats[c, cell, :] · sel_feats[cell, j, :]
        k_vec = torch.einsum("cnc_dim,ncj->cnj",
                             cand_feats.float(),
                             sel_feats[:, :m, :].float().unsqueeze(0).expand(n_cand, -1, -1, -1).reshape(n_cand, n_cells, m, C).transpose(1, 2).reshape(n_cand * m, n_cells, C)
                             ) if False else None  # placeholder — use loop below

        # More memory-efficient: batch over candidates
        # k_vec: (n_cand, n_cells, m)
        sel_f = sel_feats[:, :m, :].float()  # (n_cells, m, C)
        cand_f = cand_feats.float()  # (n_cand, n_cells, C)

        # Compute k_vec via batched matmul
        # sel_f transposed: (n_cells, C, m)
        # cand_f: (n_cand, n_cells, C)
        # k_vec[c, cell, j] = sum_d cand_f[c, cell, d] * sel_f[cell, j, d]
        # = (n_cand, n_cells, C) @ (n_cells, C, m) — need broadcast
        # Use einsum: "ncD, sDm -> ncsm" but we want per-cell
        k_vec = torch.einsum("nsd,smd->nsm", cand_f, sel_f)  # (n_cand, n_cells, m)

        # Solve L z = k_vec for each cell, using triangular solve
        # L is (n_cells, m, m), lower triangular
        # k_vec is (n_cand, n_cells, m)
        L_m = L[:, :m, :m]  # (n_cells, m, m)

        # torch.linalg.solve_triangular: (n_cells, m, m) \ (n_cells, m, n_cand)
        # Reshape k_vec to (n_cells, m, n_cand) for batch solve
        k_vec_t = k_vec.permute(1, 2, 0)  # (n_cells, m, n_cand)
        z = torch.linalg.solve_triangular(L_m, k_vec_t, upper=False)  # (n_cells, m, n_cand)

        # ||z||^2 per cell per candidate
        z_sq = (z * z).sum(dim=1)  # (n_cells, n_cand)

        # Schur complement gain per cell per candidate
        gain_per_cell = k_self.T - z_sq  # (n_cells, n_cand)
        gain_per_cell = gain_per_cell.clamp(min=eps)

        # Total gain = sum of log gains across cells
        total_gain = torch.log(gain_per_cell).sum(dim=0)  # (n_cand,)

        # Pick best candidate
        best_idx = total_gain.argmax().item()
        best_frame = candidates[best_idx]
        selected.append(best_frame)
        remaining.discard(best_frame)

        # Update Cholesky factor L for the new selection
        # New row of L: l_new = L^{-1} k_vec_best, then l_self = sqrt(gain)
        best_k_vec = k_vec[best_idx]  # (n_cells, m)
        best_k_vec_t = best_k_vec.unsqueeze(-1)  # (n_cells, m, 1)
        z_best = torch.linalg.solve_triangular(L_m, best_k_vec_t, upper=False)  # (n_cells, m, 1)
        z_best = z_best.squeeze(-1)  # (n_cells, m)

        l_self = gain_per_cell[:, best_idx].sqrt()  # (n_cells,)

        # Update L: add new row
        L[:, m, :m] = z_best
        L[:, m, m] = l_self

        # Store new selected features
        sel_feats[:, m, :] = feats[best_frame].to(sel_feats.dtype)

        m += 1
        print(f"    Step {step}/{k-1}: selected frame {best_frame} "
              f"(gain={total_gain[best_idx].item():.2f})")

    return selected


def seed_frame_multilayer(feature_list: List[torch.Tensor]) -> int:
    """Pick seed frame with highest diversity across all layers."""
    N = feature_list[0].shape[0]
    diversities = np.zeros(N)
    for features in feature_list:
        for i in range(N):
            std_per_channel = features[i].std(dim=(0, 1))
            diversities[i] += std_per_channel.mean().item()
    best = int(np.argmax(diversities))
    print(f"  Seed frame (multilayer): {best} (diversity={diversities[best]:.4f})")
    return best


@timeit
def greedy_entropy_select_multilayer(
    feature_list: List[torch.Tensor],
    k: int,
    eps: float = 1e-6,
) -> List[int]:
    """Multi-layer greedy entropy selection (LPIPS-style).

    Sums per-cell log-det gains across all feature layers.
    Each layer maintains independent Cholesky factors.

    Args:
        feature_list: List of (N, H_l, W_l, C_l) tensors (one per layer)
        k: number of frames to select
        eps: nugget for numerical stability
    """
    n_layers = len(feature_list)
    N = feature_list[0].shape[0]

    # Prepare per-layer data structures
    layer_data = []
    for features in feature_list:
        N_l, H, W, C = features.shape
        n_cells = H * W
        feats = features.reshape(N_l, n_cells, C).to(device)
        layer_data.append({
            "feats": feats,
            "n_cells": n_cells,
            "C": C,
            "sel_feats": None,
            "L": None,
        })

    # Seed
    seed = seed_frame_multilayer(feature_list)
    selected = [seed]
    remaining = set(range(N)) - {seed}

    # Initialize Cholesky for each layer
    for ld in layer_data:
        feats = ld["feats"]
        n_cells = ld["n_cells"]
        C = ld["C"]
        seed_feats = feats[seed]
        k_self_seed = (seed_feats * seed_feats).sum(dim=-1) + eps
        L_diag = k_self_seed.sqrt()

        sel_feats = torch.zeros(n_cells, k, C, device=device, dtype=feats.dtype)
        sel_feats[:, 0, :] = seed_feats

        L = torch.zeros(n_cells, k, k, device=device, dtype=torch.float32)
        L[:, 0, 0] = L_diag.float()

        ld["sel_feats"] = sel_feats
        ld["L"] = L

    m = 1

    for step in range(1, k):
        candidates = list(remaining)
        if not candidates:
            break
        n_cand = len(candidates)

        # Accumulate gains across layers
        total_gain = torch.zeros(n_cand, device=device)

        for ld in layer_data:
            feats = ld["feats"]
            sel_f = ld["sel_feats"][:, :m, :].float()
            L_m = ld["L"][:, :m, :m]

            cand_feats = feats[candidates].float()
            k_self = (cand_feats * cand_feats).sum(dim=-1).float() + eps
            k_vec = torch.einsum("nsd,smd->nsm", cand_feats, sel_f)

            k_vec_t = k_vec.permute(1, 2, 0)
            z = torch.linalg.solve_triangular(L_m, k_vec_t, upper=False)
            z_sq = (z * z).sum(dim=1)

            gain_per_cell = (k_self.T - z_sq).clamp(min=eps)
            total_gain += torch.log(gain_per_cell).sum(dim=0)

            # Store gain_per_cell for Cholesky update later
            ld["_gain_per_cell"] = gain_per_cell
            ld["_k_vec"] = k_vec

        best_idx = total_gain.argmax().item()
        best_frame = candidates[best_idx]
        selected.append(best_frame)
        remaining.discard(best_frame)

        # Update Cholesky for each layer
        for ld in layer_data:
            L_m = ld["L"][:, :m, :m]
            best_k_vec = ld["_k_vec"][best_idx].unsqueeze(-1)
            z_best = torch.linalg.solve_triangular(L_m, best_k_vec, upper=False).squeeze(-1)
            l_self = ld["_gain_per_cell"][:, best_idx].sqrt()

            ld["L"][:, m, :m] = z_best
            ld["L"][:, m, m] = l_self
            ld["sel_feats"][:, m, :] = ld["feats"][best_frame].to(ld["sel_feats"].dtype)

            del ld["_gain_per_cell"], ld["_k_vec"]

        m += 1
        print(f"    Step {step}/{k-1}: selected frame {best_frame} "
              f"(gain={total_gain[best_idx].item():.2f})")

    return selected


# ═══════════════════════════════════════════════════════
# FVS Baseline (Feature-Space)
# ═══════════════════════════════════════════════════════

@timeit
def fvs_baseline(features: torch.Tensor, k: int) -> List[int]:
    """Furthest-View Selection using cosine distance on global features.

    Global feature = mean-pooled spatial features per image.
    """
    # Mean pool: (N, H, W, C) -> (N, C)
    global_feats = features.mean(dim=(1, 2))  # (N, C)
    global_feats = F.normalize(global_feats, dim=-1)  # L2 normalize
    N = global_feats.shape[0]

    # Cosine distance matrix
    sim = global_feats @ global_feats.T  # (N, N)
    dist = 1.0 - sim  # cosine distance

    # Seed: frame 42 % N (deterministic, same as pose-based FVS)
    seed = 42 % N
    selected = [seed]
    remaining = set(range(N)) - {seed}

    for _ in range(k - 1):
        if not remaining:
            break
        candidates = list(remaining)
        # Min distance to any selected frame
        sel_dists = dist[candidates][:, selected]  # (n_cand, n_sel)
        min_dists = sel_dists.min(dim=1).values  # (n_cand,)
        best = candidates[min_dists.argmax().item()]
        selected.append(best)
        remaining.discard(best)

    print(f"  FVS selected: {sorted(selected)}")
    return selected


@timeit
def fvs_lpips_baseline(image_paths: List[str], k: int, batch_size: int = 4) -> List[int]:
    """FVS using LPIPS (VGG) perceptual distance.

    Computes pairwise LPIPS distances between all images, then
    greedily selects frames maximizing min-distance to selected set.
    """
    import lpips
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net='vgg').to(device).eval()
    N = len(image_paths)

    # Load and preprocess images to [-1, 1] range (LPIPS convention)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 1]
    ])

    imgs = []
    for p in image_paths:
        img = transform(Image.open(p).convert("RGB"))
        imgs.append(img * 2 - 1)  # scale to [-1, 1]
    imgs = torch.stack(imgs)  # (N, 3, 224, 224)

    # Compute pairwise LPIPS distance matrix
    print(f"  Computing {N}x{N} LPIPS distance matrix...")
    dist = torch.zeros(N, N)
    with torch.no_grad():
        for i in range(N):
            img_i = imgs[i:i+1].to(device).expand(min(batch_size, N), -1, -1, -1)
            for j_start in range(i + 1, N, batch_size):
                j_end = min(j_start + batch_size, N)
                batch_j = imgs[j_start:j_end].to(device)
                bs = batch_j.shape[0]
                d = loss_fn(img_i[:bs], batch_j).squeeze()
                if bs == 1:
                    dist[i, j_start] = d.item()
                    dist[j_start, i] = d.item()
                else:
                    dist[i, j_start:j_end] = d.cpu()
                    dist[j_start:j_end, i] = d.cpu()

    # FVS: greedy max-min-distance
    seed = 42 % N
    selected = [seed]
    remaining = set(range(N)) - {seed}

    for _ in range(k - 1):
        if not remaining:
            break
        candidates = list(remaining)
        sel_dists = dist[candidates][:, selected]
        min_dists = sel_dists.min(dim=1).values
        best = candidates[min_dists.argmax().item()]
        selected.append(best)
        remaining.discard(best)

    print(f"  LPIPS-FVS selected: {sorted(selected)}")
    return selected


# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# Pose-Based FVS Baselines
# ═══════════════════════════════════════════════════════

def read_colmap_images_binary(path):
    import struct, collections
    CImg = collections.namedtuple("CImg", ["qvec", "tvec", "name"])
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
            f.read(num_pts * 24)
            images[img_id] = CImg(qvec, tvec, name)
    return images

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def get_camera_poses(data_dir, scene, max_images=0):
    images_bin = os.path.join(data_dir, scene, "sparse", "0", "images.bin")
    images = read_colmap_images_binary(images_bin)
    sorted_imgs = sorted(images.values(), key=lambda x: x.name)
    if max_images > 0 and len(sorted_imgs) > max_images:
        sorted_imgs = sorted_imgs[:max_images]
    centers, directions = [], []
    for img in sorted_imgs:
        R = qvec2rotmat(img.qvec)
        C = -R.T @ img.tvec
        d = R[2, :]
        centers.append(C)
        directions.append(d / np.linalg.norm(d))
    return np.array(centers), np.array(directions)

def fvs_greedy_np(dist_matrix, k, seed=42):
    N = dist_matrix.shape[0]
    seed_idx = seed % N
    selected = [seed_idx]
    remaining = set(range(N)) - {seed_idx}
    for _ in range(k - 1):
        if not remaining:
            break
        candidates = list(remaining)
        sel_dists = dist_matrix[np.ix_(candidates, selected)]
        min_dists = sel_dists.min(axis=1)
        best = candidates[int(np.argmax(min_dists))]
        selected.append(best)
        remaining.discard(best)
    return selected

@timeit
def fvs_euclidean_baseline(data_dir, scene, k, max_images=0):
    centers, _ = get_camera_poses(data_dir, scene, max_images)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    selected = fvs_greedy_np(dist, k)
    print(f"  FVS-Euclidean selected: {sorted(selected)}")
    return selected

@timeit
def fvs_angular_baseline(data_dir, scene, k, max_images=0):
    _, directions = get_camera_poses(data_dir, scene, max_images)
    cos_sim = np.clip(directions @ directions.T, -1, 1)
    dist = np.arccos(cos_sim)
    selected = fvs_greedy_np(dist, k)
    print(f"  FVS-Angular selected: {sorted(selected)}")
    return selected

@timeit
def fvs_plucker_baseline(data_dir, scene, k, max_images=0):
    centers, directions = get_camera_poses(data_dir, scene, max_images)
    moments = np.cross(centers, directions)
    pluckers = np.concatenate([directions, moments], axis=1)
    diff = pluckers[:, None, :] - pluckers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    selected = fvs_greedy_np(dist, k)
    print(f"  FVS-Plucker selected: {sorted(selected)}")
    return selected


# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="InfoMax3D frame selection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--strategies", type=str, default="infomax,fvs",
                        help="Comma-separated: infomax, fvs, lpips_fvs")
    parser.add_argument("--multilayer_mode", type=str, default="concat",
                        choices=["concat", "sum"],
                        help="Multi-layer combination: concat (principled) or sum (independent)")
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["dinov2", "resnet"],
                        help="Feature backbone: dinov2 or resnet")
    parser.add_argument("--dino_layer", type=int, default=8,
                        help="DINOv2 intermediate layer (0-11), single layer mode")
    parser.add_argument("--dino_layers", type=str, default=None,
                        help="Comma-separated DINOv2 layers for multi-layer mode (e.g. 4,6,8)")
    parser.add_argument("--resnet_stage", type=int, default=3,
                        help="ResNet stage (1-4), single layer mode")
    parser.add_argument("--resnet_stages", type=str, default=None,
                        help="Comma-separated ResNet stages for multi-layer mode (e.g. 2,3,4)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    scene_dir = os.path.join(args.data_dir, args.scene)

    # ── Load images ──
    print("=== Loading images ===")
    image_paths = load_image_paths(scene_dir, args.data_factor, args.max_images)
    N = len(image_paths)
    print(f"  {N} images")

    # ── Determine if multi-layer mode ──
    multilayer = False
    if args.backbone == "dinov2" and args.dino_layers:
        multilayer = True
        layer_list = [int(x) for x in args.dino_layers.split(",")]
        layer_tag = "L" + "+".join(str(x) for x in layer_list)
    elif args.backbone == "resnet" and args.resnet_stages:
        multilayer = True
        layer_list = [int(x) for x in args.resnet_stages.split(",")]
        layer_tag = "S" + "+".join(str(x) for x in layer_list)
    else:
        layer_list = None
        layer_tag = None

    # ── Extract features ──
    feature_list = None
    features = None
    features_concat = None

    if multilayer:
        print(f"=== Extracting multi-layer features ({args.backbone} {layer_tag}) ===")
        if args.backbone == "dinov2":
            feature_list = extract_features_dinov2_multilayer(
                image_paths, layers=layer_list, batch_size=args.batch_size
            )
        else:
            feature_list = extract_features_resnet_multilayer(
                image_paths, stages=layer_list, batch_size=args.batch_size
            )

        # Concat mode: concatenate features per cell (only for same-grid-size layers)
        if args.multilayer_mode == "concat":
            # Check all layers have same spatial dimensions
            shapes = [f.shape[1:3] for f in feature_list]
            if len(set(shapes)) == 1:
                features_concat = torch.cat(feature_list, dim=-1)  # (N, H, W, C_total)
                features_concat = F.normalize(features_concat, dim=-1)
                print(f"  Concat feature shape: {features_concat.shape}")
            else:
                print(f"  WARNING: Layers have different spatial dims {shapes}, falling back to sum mode")
                args.multilayer_mode = "sum"

        # For FVS baseline, use concatenated mean-pooled features
        features_for_fvs = torch.cat(
            [f.mean(dim=(1, 2)) for f in feature_list], dim=-1
        )
        features_for_fvs = features_for_fvs.unsqueeze(1).unsqueeze(1)  # (N,1,1,C_total)
    else:
        print(f"=== Extracting features ({args.backbone}) ===")
        if args.backbone == "dinov2":
            features = extract_features_dinov2(
                image_paths, layer=args.dino_layer, batch_size=args.batch_size
            )
        else:
            features = extract_features_resnet(
                image_paths, stage=args.resnet_stage, batch_size=args.batch_size
            )

    # ── Determine strategies ──
    strategies = [s.strip() for s in args.strategies.split(",")]

    # ── Run each strategy ──
    results = {}
    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"=== Strategy: {strat} (k={args.k}) ===")
        print(f"{'='*60}")

        if strat == "infomax":
            if multilayer:
                if args.multilayer_mode == "concat" and features_concat is not None:
                    selected = greedy_entropy_select(features_concat, k=args.k)
                else:
                    selected = greedy_entropy_select_multilayer(feature_list, k=args.k)
            else:
                selected = greedy_entropy_select(features, k=args.k)
        elif strat == "fvs":
            if multilayer:
                selected = fvs_baseline(features_for_fvs, k=args.k)
            else:
                selected = fvs_baseline(features, k=args.k)
        elif strat == "lpips_fvs":
            selected = fvs_lpips_baseline(image_paths, k=args.k, batch_size=args.batch_size)
        elif strat == "random":
            random.seed(args.seed)
            selected = random.sample(range(N), args.k)
        elif strat == "fvs_euclidean":
            selected = fvs_euclidean_baseline(args.data_dir, args.scene, args.k, max_images=args.max_images or 0)
        elif strat == "fvs_angular":
            selected = fvs_angular_baseline(args.data_dir, args.scene, args.k, max_images=args.max_images or 0)
        elif strat == "fvs_plucker":
            selected = fvs_plucker_baseline(args.data_dir, args.scene, args.k, max_images=args.max_images or 0)
        else:
            print(f"  Unknown strategy: {strat}, skipping")
            continue

        sel_sorted = sorted(selected)
        results[strat] = sel_sorted
        print(f"  Selected: {sel_sorted}")

        # Save output
        os.makedirs(args.output_dir, exist_ok=True)
        output = {
            "scene": args.scene,
            "strategy": strat,
            "k": args.k,
            "selected_indices": sel_sorted,
            "n_images": N,
            "backbone": args.backbone,
            "layers": layer_list if multilayer else (
                [args.dino_layer] if args.backbone == "dinov2" else [args.resnet_stage]
            ),
            "multilayer_mode": args.multilayer_mode if multilayer else "single",
        }
        out_path = os.path.join(
            args.output_dir, f"train_indices_{args.scene}_{strat}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("=== SELECTION SUMMARY ===")
    for strat, sel in results.items():
        print(f"  {strat:20s}: {sel}")
    print(f"\nTimings: {json.dumps({k: f'{v:.1f}s' for k, v in time_taken.items()})}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

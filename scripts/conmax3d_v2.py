"""
ConMax3D v2: Geometry-Aware Concept Maximization for Frame Selection.

Key idea: Build "3D concepts" by linking same-semantic-concept masks across
images using centroid-epipolar distance. Connected components of matching
masks form 3D concepts. Greedy selection maximizes 3D concept coverage.

Pipeline:
  1. SAM2 masks + EfficientNet embeddings (same as v1)
  2. K-Means clustering in embedding space → semantic concepts (auto-K via elbow)
  3. Centroid-epipolar distance between same-semantic-concept masks across images
     → correspondence graph → connected components → 3D concepts
  4. Greedy selection: U(S, i) = |{3D concepts in i NOT covered by S}|
     Falls back to FVS when utility = 0.

Usage:
  python conmax3d_v2.py \
    --data_dir /path/to/dataset --scene fern \
    --output_dir /path/to/results --k 10
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from functools import wraps
from typing import List, Dict, Tuple, Set
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_utils import (
    load_poses_auto,
    compute_intrinsic_matrix,
    compute_fundamental_matrix,
    symmetric_epipolar_distance,
)

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
# Data structures
# ═══════════════════════════════════════════════════════

@dataclass
class MaskInfo:
    image_idx: int
    binary_mask: np.ndarray      # (H, W) bool
    centroid: np.ndarray         # (2,) as (row, col)
    embedding: np.ndarray        # (C,) EfficientNet feature, L2-normalized
    semantic_concept_id: int = -1
    concept_3d_id: int = -1      # assigned after correspondence graph


# ═══════════════════════════════════════════════════════
# Step 1: SAM2 mask generation + EfficientNet embeddings
# ═══════════════════════════════════════════════════════

@timeit
def generate_masks_and_features(
    images: List[np.ndarray],
    sam2_checkpoint: str,
    sam2_model_cfg: str,
    pred_iou_thresh: float = 0.8,
    efficientnet_model: str = "efficientnet_b0",
    embedding_batch_size: int = 16,
) -> List[List[MaskInfo]]:
    """Generate SAM2 masks, then extract EfficientNet embeddings per mask crop."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from img2vec_pytorch import Img2Vec

    sam2_model = build_sam2(
        sam2_model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False
    )
    mask_gen = SAM2AutomaticMaskGenerator(
        model=sam2_model, pred_iou_thresh=pred_iou_thresh
    )
    raw_masks = []
    for img in tqdm(images, desc="SAM2 mask generation"):
        raw_masks.append(mask_gen.generate(img))
    del mask_gen, sam2_model
    torch.cuda.empty_cache()

    # Phase 2: crop masks and generate EfficientNet embeddings
    img2vec = Img2Vec(cuda=torch.cuda.is_available(), model=efficientnet_model)

    all_masks: List[List[MaskInfo]] = []
    all_crops: List[Image.Image] = []
    crop_to_location: List[Tuple[int, int]] = []

    for img_idx in tqdm(range(len(images)), desc="Cropping masks"):
        img = images[img_idx]
        H, W = img.shape[:2]
        min_pixels = np.sqrt(H * W)

        image_masks = []
        for mask_data in raw_masks[img_idx]:
            seg = mask_data["segmentation"]
            if seg.sum() < min_pixels:
                continue

            mask_3d = np.dstack([seg] * 3)
            cropped_img = Image.fromarray(img * mask_3d)

            rows, cols = np.where(seg)
            centroid = np.array([rows.mean(), cols.mean()])

            mask_info = MaskInfo(
                image_idx=img_idx,
                binary_mask=seg,
                centroid=centroid,
                embedding=np.zeros(0),
            )
            image_masks.append(mask_info)
            all_crops.append(cropped_img)
            crop_to_location.append((img_idx, len(image_masks) - 1))

        all_masks.append(image_masks)

    # Batch EfficientNet embedding extraction
    print(f"  Generating EfficientNet embeddings for {len(all_crops)} crops...")
    all_embeddings = []
    for i in tqdm(range(0, len(all_crops), embedding_batch_size), desc="EfficientNet"):
        batch = all_crops[i : i + embedding_batch_size]
        batch_vecs = img2vec.get_vec(batch)
        all_embeddings.extend(batch_vecs)
    all_embeddings = np.array(all_embeddings)

    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
    all_embeddings = all_embeddings / norms

    for idx, (img_idx, mask_idx) in enumerate(crop_to_location):
        all_masks[img_idx][mask_idx].embedding = all_embeddings[idx]

    del img2vec, all_crops
    torch.cuda.empty_cache()

    total = sum(len(m) for m in all_masks)
    print(f"  Total masks after filtering: {total}")
    return all_masks


# ═══════════════════════════════════════════════════════
# Step 2: K-Means semantic concept discovery
# ═══════════════════════════════════════════════════════

@timeit
def kmeans_concept_discovery(
    all_masks: List[List[MaskInfo]], n_concepts: int = 0
) -> int:
    """K-Means clustering with elbow method to auto-detect optimal K.

    If n_concepts > 0, uses that directly. Otherwise auto-detects via elbow.
    Returns the number of concepts used.
    """
    from sklearn.cluster import KMeans

    flat = [m for img_masks in all_masks for m in img_masks]
    if not flat:
        return 0

    embeddings = np.stack([m.embedding for m in flat])
    M = len(embeddings)

    if n_concepts > 0:
        # Fixed K
        optimal_k = min(n_concepts, M)
    else:
        # Auto-detect K via elbow method
        k_range = list(range(5, min(61, M // 3), 5))
        if len(k_range) < 3:
            k_range = list(range(2, min(20, M), 2))

        inertias = []
        print(f"  Elbow search: K = {k_range[0]}..{k_range[-1]}")
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            km.fit(embeddings)
            inertias.append(km.inertia_)

        # Elbow = point of maximum curvature (second derivative)
        inertias = np.array(inertias)
        if len(inertias) >= 3:
            d1 = np.diff(inertias)       # first derivative (negative)
            d2 = np.diff(d1)             # second derivative (positive at elbow)
            elbow_idx = int(np.argmax(d2)) + 1  # +1 because diff shifts index
            optimal_k = k_range[elbow_idx]
        else:
            optimal_k = k_range[len(k_range) // 2]

        print(f"  Elbow detected at K={optimal_k} "
              f"(inertias: {', '.join(f'{k}:{i:.0f}' for k, i in zip(k_range, inertias))})")

    # Final K-Means with optimal K
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    assignments = km.fit_predict(embeddings)

    for i, mask in enumerate(flat):
        mask.semantic_concept_id = int(assignments[i])

    n_per = np.bincount(assignments, minlength=optimal_k)
    print(f"  {optimal_k} semantic concepts (K-Means), masks/concept: "
          f"min={n_per.min()} max={n_per.max()} mean={n_per.mean():.1f}")
    return optimal_k


# ═══════════════════════════════════════════════════════
# Step 3: Build 3D concepts via centroid-epipolar correspondence
# ═══════════════════════════════════════════════════════

@timeit
def compute_adaptive_threshold(
    all_masks: List[List[MaskInfo]],
    c2w: np.ndarray,
    K: np.ndarray,
    n_candidates: int = 12,
) -> float:
    """Find the threshold that produces the most useful concept graph for frame selection.

    Instead of a fixed percentile, sweeps candidate thresholds and picks
    the one that maximizes 'discriminative' 3D concepts — concepts that
    appear in >=2 but not all images. These are concepts that force the
    greedy selector to pick diverse viewpoints.

    Too tight → all singletons, concepts can't differentiate frames.
    Too loose → one giant component, all frames look the same.
    Sweet spot → many concepts appearing in subsets of images.
    """
    flat = [m for img_masks in all_masks for m in img_masks]
    M = len(flat)
    N = len(all_masks)
    if M == 0:
        return 20.0

    # Group masks by semantic concept
    concept_masks: Dict[int, List[int]] = {}
    for i, m in enumerate(flat):
        concept_masks.setdefault(m.semantic_concept_id, []).append(i)

    # Compute ALL same-semantic-concept cross-image mask pair distances
    F_cache: Dict[Tuple[int, int], np.ndarray] = {}
    edges: List[Tuple[int, int, float]] = []

    for sem_id, mask_indices in concept_masks.items():
        img_to_masks: Dict[int, List[int]] = {}
        for mi in mask_indices:
            img_to_masks.setdefault(flat[mi].image_idx, []).append(mi)

        img_list = sorted(img_to_masks.keys())
        for a_idx in range(len(img_list)):
            img_a = img_list[a_idx]
            for b_idx in range(a_idx + 1, len(img_list)):
                img_b = img_list[b_idx]

                key = (img_a, img_b)
                if key not in F_cache:
                    F_cache[key] = compute_fundamental_matrix(
                        c2w[img_a], c2w[img_b], K
                    )
                F_ab = F_cache[key]

                for mi_a in img_to_masks[img_a]:
                    for mi_b in img_to_masks[img_b]:
                        d = symmetric_epipolar_distance(
                            flat[mi_a].centroid, flat[mi_b].centroid, F_ab,
                        )
                        edges.append((mi_a, mi_b, d))

    if not edges:
        return 20.0

    dists = np.array([e[2] for e in edges])
    print(f"  {len(edges)} mask pairs, {len(F_cache)} F matrices")
    print(f"  Distance distribution: min={dists.min():.2f} "
          f"median={np.median(dists):.2f} max={dists.max():.2f}")

    # Sweep candidate thresholds sampled from distance distribution
    percentiles = np.linspace(5, 60, n_candidates)
    candidates = np.unique(np.percentile(dists, percentiles))

    best_thresh = float(candidates[0])
    best_score = -1
    best_stats = ""

    for thresh in candidates:
        # Build adjacency matrix at this threshold
        adj = lil_matrix((M, M), dtype=bool)
        for mi_a, mi_b, d in edges:
            if d < thresh:
                adj[mi_a, mi_b] = True
                adj[mi_b, mi_a] = True

        n_comp, labels = connected_components(adj.tocsr(), directed=False)

        # Count which images each concept appears in
        concept_images: Dict[int, Set[int]] = {}
        for i, m in enumerate(flat):
            concept_images.setdefault(int(labels[i]), set()).add(m.image_idx)

        # Discriminative concepts: appear in >=2 but <N images
        n_discriminative = sum(
            1 for imgs in concept_images.values()
            if 2 <= len(imgs) < N
        )
        n_multi = sum(1 for imgs in concept_images.values() if len(imgs) >= 2)
        n_singletons = sum(1 for imgs in concept_images.values() if len(imgs) == 1)

        # Maximize discriminative concepts; tiebreak by total multi-image concepts
        score = n_discriminative + 0.001 * n_multi

        stats = (f"thresh={thresh:6.2f}px: {n_comp:4d} concepts, "
                 f"{n_discriminative:3d} discriminative, "
                 f"{n_multi:3d} multi-img, {n_singletons:4d} singletons")
        print(f"    {stats}")

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            best_stats = stats

    print(f"  -> Best: {best_stats}")
    return max(best_thresh, 1.0)


@timeit
def build_3d_concepts(
    all_masks: List[List[MaskInfo]],
    c2w: np.ndarray,
    K: np.ndarray,
    epipolar_threshold: float = 20.0,
) -> int:
    """Link same-semantic-concept masks across images using centroid-epipolar distance.

    Builds a correspondence graph where nodes are masks and edges connect masks
    that (a) share a semantic concept and (b) have symmetric epipolar distance
    below the threshold. Connected components become 3D concepts.

    If epipolar_threshold <= 0, auto-detect from the distance distribution.

    Returns the number of 3D concepts found.
    """
    flat = [m for img_masks in all_masks for m in img_masks]
    M = len(flat)
    if M == 0:
        return 0

    # Auto-detect threshold if requested
    if epipolar_threshold <= 0:
        epipolar_threshold = compute_adaptive_threshold(all_masks, c2w, K)

    # Group masks by semantic concept
    concept_masks: Dict[int, List[int]] = {}
    for i, m in enumerate(flat):
        concept_masks.setdefault(m.semantic_concept_id, []).append(i)

    # Build sparse adjacency matrix
    adj = lil_matrix((M, M), dtype=bool)

    # Cache fundamental matrices
    F_cache: Dict[Tuple[int, int], np.ndarray] = {}

    n_pairs_checked = 0
    n_edges = 0

    for sem_id, mask_indices in concept_masks.items():
        # Group by image within this semantic concept
        img_to_masks: Dict[int, List[int]] = {}
        for mi in mask_indices:
            img_to_masks.setdefault(flat[mi].image_idx, []).append(mi)

        img_list = sorted(img_to_masks.keys())

        # Check all cross-image pairs within this semantic concept
        for a_idx in range(len(img_list)):
            img_a = img_list[a_idx]
            for b_idx in range(a_idx + 1, len(img_list)):
                img_b = img_list[b_idx]

                key = (img_a, img_b)
                if key not in F_cache:
                    F_cache[key] = compute_fundamental_matrix(
                        c2w[img_a], c2w[img_b], K
                    )
                F_ab = F_cache[key]

                for mi_a in img_to_masks[img_a]:
                    for mi_b in img_to_masks[img_b]:
                        n_pairs_checked += 1
                        d = symmetric_epipolar_distance(
                            flat[mi_a].centroid,
                            flat[mi_b].centroid,
                            F_ab,
                        )
                        if d < epipolar_threshold:
                            adj[mi_a, mi_b] = True
                            adj[mi_b, mi_a] = True
                            n_edges += 1

    print(f"  Checked {n_pairs_checked} mask pairs, {n_edges} edges, "
          f"{len(F_cache)} F matrices (threshold={epipolar_threshold:.2f}px)")

    # Connected components → 3D concepts
    n_components, labels = connected_components(adj.tocsr(), directed=False)

    for i, m in enumerate(flat):
        m.concept_3d_id = int(labels[i])

    sizes = np.bincount(labels)
    n_singletons = np.sum(sizes == 1)
    print(f"  {n_components} 3D concepts ({n_singletons} singletons), "
          f"size: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.1f}")

    return n_components, epipolar_threshold


# ═══════════════════════════════════════════════════════
# Step 4: Greedy selection maximizing 3D concept coverage
# ═══════════════════════════════════════════════════════

def _fvs_pick(cam_positions, selected_list, candidate_set):
    """Pick candidate with max min-distance to selected set.

    First call (empty selected_list): pick random frame with seed=42
    to match reference FVS implementation.
    """
    if not selected_list:
        # Match reference baselines.py: random start with seed=42
        rng = np.random.RandomState(42)
        N = len(cam_positions)
        first = rng.randint(N)
        # If first is not in candidate_set (e.g., filtered), pick closest candidate
        if first in candidate_set:
            return first
        # Fallback: pick the candidate closest to the random pick
        return min(candidate_set,
                   key=lambda i: np.linalg.norm(cam_positions[i] - cam_positions[first]))
    sel_pos = cam_positions[selected_list]
    best_img, best_dist = None, -1.0
    for cand in candidate_set:
        d = float(np.min(np.linalg.norm(sel_pos - cam_positions[cand], axis=1)))
        if d > best_dist:
            best_dist = d
            best_img = cand
    return best_img


def _get_image_concepts(all_masks):
    return {i: {m.concept_3d_id for m in masks} for i, masks in enumerate(all_masks)}


@timeit
def strategy_3d_concepts(all_masks, cam_positions, k):
    """Pure 3D concept maximization with FVS fallback."""
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)
    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    for step in range(k):
        best_img, best_score = None, -1
        for cand in remaining:
            n_new = len(image_concepts.get(cand, set()) - covered)
            if n_new > best_score:
                best_score = n_new
                best_img = cand

        if best_score > 0 and best_img is not None:
            selected.append(best_img)
            remaining.remove(best_img)
            new_c = image_concepts.get(best_img, set()) - covered
            covered.update(new_c)
            print(f"  Step {step+1}/{k}: img {best_img} (+{best_score} concepts, covered={len(covered)})")
        else:
            pick = _fvs_pick(cam_positions, selected, remaining)
            selected.append(pick)
            remaining.remove(pick)
            print(f"  Step {step+1}/{k}: img {pick} (FVS fallback)")
    return selected


@timeit
def strategy_concept_filtered_fvs(all_masks, cam_positions, k):
    """FVS restricted to candidates that bring new 3D concepts.

    At each step:
    - Filter candidates to those with uncovered concepts
    - Among those, pick the one farthest from selected set (FVS)
    - If all concepts covered, pure FVS fallback
    """
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)
    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    for step in range(k):
        with_new = [c for c in remaining if len(image_concepts.get(c, set()) - covered) > 0]

        if with_new:
            pick = _fvs_pick(cam_positions, selected, with_new)
            selected.append(pick)
            remaining.remove(pick)
            new_c = image_concepts.get(pick, set()) - covered
            covered.update(new_c)
            print(f"  Step {step+1}/{k}: img {pick} (+{len(new_c)} concepts, "
                  f"FVS among {len(with_new)} candidates, covered={len(covered)})")
        else:
            pick = _fvs_pick(cam_positions, selected, remaining)
            selected.append(pick)
            remaining.remove(pick)
            print(f"  Step {step+1}/{k}: img {pick} (FVS fallback)")
    return selected


@timeit
def strategy_pure_fvs(cam_positions, k):
    """Pure FVS baseline (no concepts)."""
    N = len(cam_positions)
    selected, remaining = [], set(range(N))

    for step in range(k):
        pick = _fvs_pick(cam_positions, selected, remaining)
        selected.append(pick)
        remaining.remove(pick)
        if step < 3 or step == k - 1:
            print(f"  Step {step+1}/{k}: img {pick}")
    return selected


@timeit
def strategy_product_score(all_masks, cam_positions, k):
    """Score = n_new_concepts × min_distance. No hyperparameters.

    Balances concept novelty and geometric diversity via multiplication.
    When concepts exhausted, falls back to pure FVS.
    """
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)
    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    # Normalize distances for fair product
    all_dists = np.linalg.norm(
        cam_positions[:, None] - cam_positions[None, :], axis=2
    )
    max_dist = all_dists.max() + 1e-8

    for step in range(k):
        best_img, best_score = None, -1.0

        for cand in remaining:
            n_new = len(image_concepts.get(cand, set()) - covered)
            if selected:
                sel_pos = cam_positions[selected]
                min_d = float(np.min(np.linalg.norm(sel_pos - cam_positions[cand], axis=1)))
            else:
                min_d = max_dist
            norm_d = min_d / max_dist

            # Product: both terms must be high
            score = (n_new + 0.01) * norm_d  # +0.01 so FVS works when n_new=0
            if score > best_score:
                best_score = score
                best_img = cand

        selected.append(best_img)
        remaining.remove(best_img)
        new_c = image_concepts.get(best_img, set()) - covered
        covered.update(new_c)
        print(f"  Step {step+1}/{k}: img {best_img} (+{len(new_c)} concepts, "
              f"score={best_score:.4f}, covered={len(covered)})")
    return selected


# ═══════════════════════════════════════════════════════
# Step 5: New strategies
# ═══════════════════════════════════════════════════════

def _compute_image_features(all_masks):
    """Compute per-image feature as L2-normalized mean of mask embeddings."""
    features = []
    dim = None
    for masks in all_masks:
        if masks:
            embs = np.stack([m.embedding for m in masks])
            feat = embs.mean(axis=0)
            if dim is None:
                dim = feat.shape[0]
        else:
            feat = np.zeros(dim or 1280)
        features.append(feat)
    features = np.stack(features)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    return features / norms


@timeit
def strategy_soft_concept_bonus(all_masks, cam_positions, k, beta=1.0):
    """FVS with soft concept bonus (no hard filtering).

    score(i) = min_dist(i, S) * (1 + beta * n_new / n_uncovered)
    Frames with new concepts get a multiplicative boost but no frame is excluded.
    """
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)
    all_concepts = set()
    for concepts in image_concepts.values():
        all_concepts.update(concepts)

    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    for step in range(k):
        if not selected:
            pick = _fvs_pick(cam_positions, selected, remaining)
        else:
            n_uncovered = max(len(all_concepts - covered), 1)
            sel_pos = cam_positions[selected]
            best_img, best_score = None, -1.0

            for cand in remaining:
                min_d = float(np.min(np.linalg.norm(
                    sel_pos - cam_positions[cand], axis=1)))
                n_new = len(image_concepts.get(cand, set()) - covered)
                score = min_d * (1.0 + beta * n_new / n_uncovered)
                if score > best_score:
                    best_score = score
                    best_img = cand
            pick = best_img

        selected.append(pick)
        remaining.remove(pick)
        new_c = image_concepts.get(pick, set()) - covered
        covered.update(new_c)
        print(f"  Step {step+1}/{k}: img {pick} (+{len(new_c)} concepts, "
              f"covered={len(covered)}/{len(all_concepts)})")
    return selected


@timeit
def strategy_joint_space_fvs(all_masks, cam_positions, k, alpha=0.5):
    """FVS in joint pose+feature space (no concept threshold needed).

    dist(i,j) = alpha * norm_pose_dist + (1-alpha) * norm_feature_dist
    Naturally spreads across both geometry AND visual content.
    """
    N = len(cam_positions)
    image_features = _compute_image_features(all_masks)

    # Normalized pose distances
    pose_dists = np.linalg.norm(
        cam_positions[:, None] - cam_positions[None, :], axis=2)
    max_pose = pose_dists.max() + 1e-8
    norm_pose = pose_dists / max_pose

    # Feature distances (1 - cosine sim, features are L2-normed)
    cos_sim = image_features @ image_features.T
    feat_dists = 1.0 - cos_sim
    max_feat = feat_dists.max() + 1e-8
    norm_feat = feat_dists / max_feat

    # Joint distance matrix
    joint = alpha * norm_pose + (1.0 - alpha) * norm_feat

    # FVS in joint space
    selected, remaining = [], set(range(N))

    # First pick: random seed=42
    rng = np.random.RandomState(42)
    first = rng.randint(N)
    selected.append(first)
    remaining.remove(first)
    print(f"  Step 1/{k}: img {first} (seed, alpha={alpha})")

    for step in range(1, k):
        best_img, best_dist = None, -1.0
        for cand in remaining:
            min_d = float(min(joint[cand, s] for s in selected))
            if min_d > best_dist:
                best_dist = min_d
                best_img = cand
        selected.append(best_img)
        remaining.remove(best_img)
        print(f"  Step {step+1}/{k}: img {best_img} (joint_min_dist={best_dist:.4f})")
    return selected


@timeit
def strategy_adaptive_concept_fvs(all_masks, cam_positions, k,
                                   coverage_threshold=0.5):
    """Concept-filtered FVS that relaxes to pure FVS as coverage grows.

    When fraction uncovered > coverage_threshold: filter to concept-bearing candidates.
    When fraction uncovered <= coverage_threshold: pure FVS (geometry wins).
    """
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)
    all_concepts = set()
    for concepts in image_concepts.values():
        all_concepts.update(concepts)
    total = max(len(all_concepts), 1)

    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    for step in range(k):
        frac_uncovered = len(all_concepts - covered) / total
        use_concepts = frac_uncovered > coverage_threshold

        if use_concepts:
            with_new = [c for c in remaining
                        if len(image_concepts.get(c, set()) - covered) > 0]
            if with_new:
                pick = _fvs_pick(cam_positions, selected, with_new)
                selected.append(pick)
                remaining.remove(pick)
                new_c = image_concepts.get(pick, set()) - covered
                covered.update(new_c)
                print(f"  Step {step+1}/{k}: img {pick} (+{len(new_c)} concepts, "
                      f"FILTERED FVS among {len(with_new)}, "
                      f"coverage={1-len(all_concepts-covered)/total:.1%})")
                continue

        # Pure FVS
        pick = _fvs_pick(cam_positions, selected, remaining)
        selected.append(pick)
        remaining.remove(pick)
        new_c = image_concepts.get(pick, set()) - covered
        covered.update(new_c)
        print(f"  Step {step+1}/{k}: img {pick} (+{len(new_c)} concepts, "
              f"PURE FVS, coverage={1-len(all_concepts-covered)/total:.1%})")
    return selected


@timeit
def strategy_submodular_geometric(all_masks, cam_positions, k):
    """Submodular concept coverage with geometric floor constraint.

    Greedy maximization of concept coverage, subject to:
    each new frame must be >= D_min from all selected frames.
    D_min = 10th percentile of all pairwise distances.
    Tie-break: farthest from selected set (FVS-style).
    """
    N = len(all_masks)
    image_concepts = _get_image_concepts(all_masks)

    # Precompute distances and D_min
    all_dists = np.linalg.norm(
        cam_positions[:, None] - cam_positions[None, :], axis=2)
    triu_dists = all_dists[np.triu_indices(N, k=1)]
    D_min = float(np.percentile(triu_dists, 10))
    print(f"  D_min = {D_min:.4f} (10th pctile of pairwise dists)")

    covered: Set[int] = set()
    selected, remaining = [], set(range(N))

    for step in range(k):
        # First pick: random seed=42
        if not selected:
            rng = np.random.RandomState(42)
            pick = rng.randint(N)
            selected.append(pick)
            remaining.remove(pick)
            new_c = image_concepts.get(pick, set()) - covered
            covered.update(new_c)
            print(f"  Step {step+1}/{k}: img {pick} (seed, +{len(new_c)} concepts)")
            continue

        # Find feasible candidates
        feasible = [c for c in remaining
                    if min(all_dists[c, s] for s in selected) >= D_min]
        relaxed = False
        if not feasible:
            feasible = list(remaining)
            relaxed = True

        # Among feasible: max new concepts, tiebreak by max min-distance
        best_img, best_gain, best_dist = None, -1, -1.0
        for cand in feasible:
            n_new = len(image_concepts.get(cand, set()) - covered)
            min_d = float(min(all_dists[cand, s] for s in selected))
            if n_new > best_gain or (n_new == best_gain and min_d > best_dist):
                best_gain = n_new
                best_dist = min_d
                best_img = cand

        selected.append(best_img)
        remaining.remove(best_img)
        new_c = image_concepts.get(best_img, set()) - covered
        covered.update(new_c)
        tag = " RELAXED" if relaxed else ""
        print(f"  Step {step+1}/{k}: img {best_img} (+{len(new_c)} concepts, "
              f"{len(feasible)} feasible{tag}, covered={len(covered)})")
    return selected


# ═══════════════════════════════════════════════════════
# Image loading
# ═══════════════════════════════════════════════════════

def load_images(scene_dir: str, N: int, H: int, W: int, factor: int) -> List[np.ndarray]:
    img_dir = os.path.join(scene_dir, f"images_{factor}")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_dir, "images")

    img_files = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    images = []
    for fname in img_files[:N]:
        img = Image.open(os.path.join(img_dir, fname))
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
        images.append(np.array(img))

    print(f"  Loaded {len(images)} images from {img_dir}")
    return images


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ConMax3D v2 frame selection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--n_concepts", type=int, default=0,
                        help="Semantic concepts (0 = auto-detect via elbow)")
    parser.add_argument("--epipolar_threshold", type=float, default=0.0,
                        help="Centroid-epipolar distance threshold in pixels. "
                             "0 = auto-detect from distance distribution")
    parser.add_argument("--strategies", type=str, default="all",
                        help="Comma-separated strategies or 'all'. Options: "
                             "3d_concepts, concept_filtered_fvs, pure_fvs, product_score")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--sam2_checkpoint", type=str,
        default="/gpfs/workdir/malhotraa/segment-anything-2/checkpoints/sam2_hiera_large.pt",
    )
    parser.add_argument("--sam2_model_cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--efficientnet_model", type=str, default="efficientnet_b0")
    args = parser.parse_args()

    scene_dir = os.path.join(args.data_dir, args.scene)

    # ── Load poses ──
    print("=== Loading camera poses ===")
    c2w, cam_pos, focal, H, W = load_poses_auto(scene_dir, factor=args.data_factor)
    K_mat = compute_intrinsic_matrix(focal, H, W)
    N = len(c2w)
    print(f"  {N} images, {H}x{W}, focal={focal:.1f}")

    # ── Optional subsampling ──
    subset_indices = None
    if args.max_images and N > args.max_images:
        stride = N / args.max_images
        subset_indices = [int(i * stride) for i in range(args.max_images)]
        print(f"  Subsampling {N} -> {len(subset_indices)} images (stride ~{stride:.1f})")
        c2w = c2w[subset_indices]
        cam_pos = cam_pos[subset_indices]
        N = len(c2w)

    # ── Load images ──
    print("=== Loading images ===")
    max_img = len(c2w) if subset_indices is None else max(subset_indices) + 1
    all_images = load_images(scene_dir, max_img, H, W, args.data_factor)
    if subset_indices is not None:
        all_images = [all_images[i] for i in subset_indices]

    # ── SAM2 masks + EfficientNet ──
    print("=== SAM2 masks + EfficientNet embeddings ===")
    all_masks = generate_masks_and_features(
        all_images, args.sam2_checkpoint, args.sam2_model_cfg,
        pred_iou_thresh=args.pred_iou_thresh,
        efficientnet_model=args.efficientnet_model,
    )

    # ── Semantic concepts via K-Means ──
    print(f"=== K-Means semantic concept discovery (n_concepts={args.n_concepts}) ===")
    n_concepts = kmeans_concept_discovery(all_masks, args.n_concepts)

    # ── 3D concepts via centroid-epipolar correspondence ──
    thresh_str = f"{args.epipolar_threshold}px" if args.epipolar_threshold > 0 else "auto"
    print(f"=== Building 3D concepts (threshold={thresh_str}) ===")
    n_3d, actual_threshold = build_3d_concepts(all_masks, c2w, K_mat, args.epipolar_threshold)

    # ── Determine strategies to run ──
    all_strategies = [
        "concept_filtered_fvs", "pure_fvs", "soft_concept_bonus",
        "joint_space_fvs", "adaptive_concept_fvs", "submodular_geometric",
        "3d_concepts", "product_score",
    ]
    if args.strategies == "all":
        strategies = all_strategies
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]

    # ── Run each strategy ──
    results = {}
    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"=== Strategy: {strat} (k={args.k}) ===")
        print(f"{'='*60}")

        if strat == "3d_concepts":
            selected = strategy_3d_concepts(all_masks, cam_pos, k=args.k)
        elif strat == "concept_filtered_fvs":
            selected = strategy_concept_filtered_fvs(all_masks, cam_pos, k=args.k)
        elif strat == "pure_fvs":
            selected = strategy_pure_fvs(cam_pos, k=args.k)
        elif strat == "product_score":
            selected = strategy_product_score(all_masks, cam_pos, k=args.k)
        elif strat == "soft_concept_bonus":
            selected = strategy_soft_concept_bonus(all_masks, cam_pos, k=args.k)
        elif strat == "joint_space_fvs":
            selected = strategy_joint_space_fvs(all_masks, cam_pos, k=args.k)
        elif strat == "adaptive_concept_fvs":
            selected = strategy_adaptive_concept_fvs(all_masks, cam_pos, k=args.k)
        elif strat == "submodular_geometric":
            selected = strategy_submodular_geometric(all_masks, cam_pos, k=args.k)
        else:
            print(f"  Unknown strategy: {strat}, skipping")
            continue

        # Map back to original indices
        if subset_indices is not None:
            sel_orig = sorted([subset_indices[i] for i in selected])
        else:
            sel_orig = sorted(selected)

        results[strat] = sel_orig
        print(f"  Selected: {sel_orig}")

        # Save indices for this strategy
        os.makedirs(args.output_dir, exist_ok=True)
        output = {
            "scene": args.scene,
            "strategy": strat,
            "k": args.k,
            "selected_indices": sel_orig,
            "n_images": N,
            "n_semantic_concepts": n_concepts,
            "n_3d_concepts": n_3d,
            "epipolar_threshold": actual_threshold,
            "total_masks": sum(len(m) for m in all_masks),
        }
        out_path = os.path.join(args.output_dir, f"train_indices_{args.scene}_{strat}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("=== SELECTION SUMMARY ===")
    for strat, sel in results.items():
        print(f"  {strat:30s}: {sel}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

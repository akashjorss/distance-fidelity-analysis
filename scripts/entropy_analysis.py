"""
Entropy Analysis: Swap-based local search & entropy-PSNR correlation.

For each scene:
1. Extract DINOv2 L4 features
2. Compute log-det entropy for all methods' selections
3. Run swap-based local search starting from greedy
4. Save results as JSON

Usage:
  python entropy_analysis.py --data_dir /path/to/LLFF --scene fern \
      --results_base /path/to/results --output_dir /path/to/output
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
print(f"Device: {device}")


# ─── Image loading (from infomax3d.py) ───

def load_image_paths(scene_dir: str, factor: int) -> List[str]:
    img_dir = os.path.join(scene_dir, f"images_{factor}")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_dir, "images")
    fnames = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    paths = [os.path.join(img_dir, f) for f in fnames]
    print(f"  Found {len(paths)} images in {img_dir}")
    return paths


# ─── Feature extraction (from infomax3d.py) ───

def extract_features_dinov2(image_paths: List[str], layer: int = 4,
                            batch_size: int = 8) -> torch.Tensor:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.to(device).eval()

    intermediate_output = {}
    def hook_fn(module, input, output):
        intermediate_output["feat"] = output
    handle = model.blocks[layer].register_forward_hook(hook_fn)

    img_size = 518
    patch_size = 14
    grid_size = img_size // patch_size  # 37

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
        imgs = imgs.to(device)

        with torch.no_grad():
            _ = model(imgs)
            feat = intermediate_output["feat"]  # (B, 1+H*W, C) with CLS token
            feat = feat[:, 1:, :]  # remove CLS: (B, H*W, C)
            feat = feat.reshape(-1, grid_size, grid_size, feat.shape[-1])
            feat = F.normalize(feat, dim=-1)
            all_features.append(feat.cpu())

    handle.remove()
    features = torch.cat(all_features, dim=0)  # (N, 37, 37, 768)
    print(f"  Features: {features.shape}")
    return features


# ─── Entropy computation ───

def compute_entropy(feats_flat: torch.Tensor, indices: List[int],
                    eps: float = 1e-6) -> float:
    """Compute total log-det entropy for a subset.
    feats_flat: (N, n_cells, C) on GPU
    indices: list of k indices
    Returns: scalar total entropy
    """
    k = len(indices)
    F_sel = feats_flat[indices]  # (k, n_cells, C)
    # Gram matrix per cell: (n_cells, k, k)
    G = torch.einsum('knd,mnd->nkm', F_sel, F_sel)
    G = G + eps * torch.eye(k, device=G.device, dtype=G.dtype).unsqueeze(0)
    # Log-det
    _, logdet = torch.linalg.slogdet(G.float())  # (n_cells,)
    return logdet.sum().item()


# ─── Swap-based local search ───

def swap_local_search(feats_flat: torch.Tensor, greedy_indices: List[int],
                      eps: float = 1e-6, max_rounds: int = 50) -> Tuple[List[int], float, int]:
    """Swap-based local search improving on greedy.
    Returns: (improved_indices, final_entropy, num_swaps)
    """
    N = feats_flat.shape[0]
    k = len(greedy_indices)
    selected = list(greedy_indices)
    current_entropy = compute_entropy(feats_flat, selected, eps)
    print(f"  Swap search: initial entropy = {current_entropy:.4f}")

    total_swaps = 0
    for round_num in range(max_rounds):
        best_gain = 0.0
        best_swap = None
        selected_set = set(selected)
        unselected = [j for j in range(N) if j not in selected_set]

        # Evaluate all k * (N-k) swaps
        # For efficiency, batch: for each position i in selected, try all unselected
        for si in range(k):
            orig_idx = selected[si]
            # Try replacing position si with each unselected index
            for uj in unselected:
                trial = selected.copy()
                trial[si] = uj
                trial_entropy = compute_entropy(feats_flat, trial, eps)
                gain = trial_entropy - current_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (si, uj, trial_entropy)

        if best_swap is None or best_gain < 1e-8:
            print(f"  Round {round_num+1}: no improving swap found. Converged.")
            break

        si, uj, new_entropy = best_swap
        old_idx = selected[si]
        selected[si] = uj
        current_entropy = new_entropy
        total_swaps += 1
        print(f"  Round {round_num+1}: swap {old_idx} -> {uj}, "
              f"gain={best_gain:.4f}, entropy={current_entropy:.4f}")

    return selected, current_entropy, total_swaps


# ─── Load indices from JSON ───

def load_indices(path: str) -> Optional[List[int]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("selected_indices", None)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Base data dir (e.g. /path/to/LLFF)")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--results_base", required=True, help="Base results dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    scene_dir = os.path.join(args.data_dir, args.scene)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Extract features
    print(f"=== {args.scene}: Feature extraction ===")
    image_paths = load_image_paths(scene_dir, args.data_factor)
    N = len(image_paths)
    features = extract_features_dinov2(image_paths, layer=args.layer,
                                       batch_size=args.batch_size)
    # Flatten spatial dims
    feats_flat = features.reshape(N, -1, features.shape[-1]).to(device)  # (N, n_cells, C)

    # 2. Load all methods' indices
    print(f"\n=== {args.scene}: Loading method indices ===")
    rb = args.results_base
    methods = {
        "infomax_greedy": f"{rb}/v3_dinov2_L4/{args.scene}/train_indices_{args.scene}_infomax.json",
        "fvs_cosine": f"{rb}/v3/{args.scene}/train_indices_{args.scene}_fvs.json",
        "fvs_plucker": f"{rb}/v3_fvs_plucker/{args.scene}/train_indices_{args.scene}_fvs_plucker.json",
        "random_s42": f"{rb}/v3_random_s42/{args.scene}/train_indices_{args.scene}_random.json",
        "fvs_euclidean": f"{rb}/v3_fvs_euclidean/{args.scene}/train_indices_{args.scene}_fvs_euclidean.json",
        "fvs_angular": f"{rb}/v3_fvs_angular/{args.scene}/train_indices_{args.scene}_fvs_angular.json",
    }

    results = {"scene": args.scene, "N": N, "k": 10, "methods": {}}

    for method_name, idx_path in methods.items():
        indices = load_indices(idx_path)
        if indices is None:
            print(f"  {method_name}: no index file at {idx_path}")
            continue
        entropy = compute_entropy(feats_flat, indices)
        results["methods"][method_name] = {
            "indices": indices,
            "entropy": entropy,
        }
        print(f"  {method_name}: indices={indices}, entropy={entropy:.4f}")

    # 3. Swap-based local search
    greedy_indices = results["methods"].get("infomax_greedy", {}).get("indices")
    if greedy_indices is None:
        print("ERROR: no greedy indices found")
        sys.exit(1)

    print(f"\n=== {args.scene}: Swap local search ===")
    t0 = time.time()
    swap_indices, swap_entropy, num_swaps = swap_local_search(
        feats_flat, greedy_indices
    )
    swap_time = time.time() - t0

    results["methods"]["infomax_swap"] = {
        "indices": sorted(swap_indices),
        "entropy": swap_entropy,
        "num_swaps": num_swaps,
        "swap_time_s": swap_time,
        "changed_from_greedy": sorted(swap_indices) != sorted(greedy_indices),
    }

    greedy_entropy = results["methods"]["infomax_greedy"]["entropy"]
    print(f"\n=== {args.scene}: Summary ===")
    print(f"  N = {N}")
    print(f"  Greedy entropy:  {greedy_entropy:.4f}  indices={sorted(greedy_indices)}")
    print(f"  Swap entropy:    {swap_entropy:.4f}  indices={sorted(swap_indices)}")
    print(f"  Δ entropy:       {swap_entropy - greedy_entropy:.4f}")
    print(f"  Num swaps:       {num_swaps}")
    print(f"  Changed:         {sorted(swap_indices) != sorted(greedy_indices)}")
    print(f"  Swap search time: {swap_time:.1f}s")

    for mname, mdata in results["methods"].items():
        if mname in ("infomax_greedy", "infomax_swap"):
            continue
        print(f"  {mname}: entropy={mdata['entropy']:.4f}")

    # 4. Save results
    out_path = os.path.join(args.output_dir, f"entropy_{args.scene}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # 5. Save swap indices in standard format (for gsplat training)
    if sorted(swap_indices) != sorted(greedy_indices):
        swap_idx_path = os.path.join(args.output_dir,
                                      f"train_indices_{args.scene}_infomax_swap.json")
        with open(swap_idx_path, "w") as f:
            json.dump({"selected_indices": sorted(swap_indices)}, f)
        print(f"Saved swap indices to {swap_idx_path}")


if __name__ == "__main__":
    main()

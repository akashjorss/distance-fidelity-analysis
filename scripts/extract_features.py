"""Extract and cache visual features for all images across all datasets.

Models:
  - DINOv2 (ViT-B/14, layers 4,8) -> [N_patches, D] per image
  - AlexNet (FC6 features + softmax) -> [4096] per image
  - CLIP (ViT-B/16 image features) -> [512] per image

Output: results/features/{dataset}/{model}/{scene}.pt

Usage:
    python extract_features.py --dataset llff
    python extract_features.py --dataset tt --model dinov2
    python extract_features.py --dataset ns --model clip
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms

WORKDIR = "/gpfs/workdir/malhotraa"
RESULTS_BASE = f"{WORKDIR}/ConMax3D_reproduce/results"
FEATURES_BASE = f"{RESULTS_BASE}/features"

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


class ImageFolderDataset(Dataset):
    """Simple dataset that loads all images from a directory."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        exts = ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG")
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, ext))))
        # Also check for test_images subdirectory (NeRF Synthetic)
        test_dir = os.path.join(os.path.dirname(image_dir), "test_images")
        if os.path.isdir(test_dir):
            for ext in exts:
                self.image_paths.extend(sorted(glob.glob(os.path.join(test_dir, ext))))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.image_paths[idx]


def extract_dinov2(image_dir, device, batch_size=8, layers=(4, 8)):
    """Extract DINOv2 features from all images."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_features = {f"layer_{l}": [] for l in layers}
    all_cls = {f"layer_{l}": [] for l in layers}
    all_paths = []

    hooks = {}
    intermediate = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            intermediate[name] = output
        return hook_fn

    for l in layers:
        hooks[f"layer_{l}"] = model.blocks[l].register_forward_hook(make_hook(f"layer_{l}"))

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            _ = model(images)

            for l in layers:
                feat = intermediate[f"layer_{l}"]
                # feat shape: [B, N_patches+1, D]
                cls_token = feat[:, 0, :]  # [B, D]
                patch_tokens = feat[:, 1:, :]  # [B, N_patches, D]
                all_cls[f"layer_{l}"].append(cls_token.cpu())
                all_features[f"layer_{l}"].append(patch_tokens.cpu())

            all_paths.extend(paths)

    for h in hooks.values():
        h.remove()

    result = {"paths": all_paths}
    for l in layers:
        key = f"layer_{l}"
        result[f"{key}_cls"] = torch.cat(all_cls[key], dim=0)  # [N_images, D]
        result[f"{key}_patches"] = torch.cat(all_features[key], dim=0)  # [N_images, N_patches, D]

    return result


def extract_alexnet(image_dir, device, batch_size=16):
    """Extract AlexNet FC6 features and softmax probabilities."""
    import torchvision.models as models

    model = models.alexnet(pretrained=True).to(device).eval()

    # Hook into FC6 (classifier[1] = ReLU after first linear)
    fc6_features = []

    def fc6_hook(module, input, output):
        fc6_features.append(output.detach())

    hook = model.classifier[1].register_forward_hook(fc6_hook)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_fc6 = []
    all_softmax = []
    all_paths = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            logits = model(images)
            softmax = F.softmax(logits, dim=1)

            all_fc6.append(torch.cat(fc6_features, dim=0).cpu())
            fc6_features.clear()
            all_softmax.append(softmax.cpu())
            all_paths.extend(paths)

    hook.remove()

    return {
        "paths": all_paths,
        "fc6": torch.cat(all_fc6, dim=0),  # [N, 4096]
        "softmax": torch.cat(all_softmax, dim=0),  # [N, 1000]
    }


def extract_clip(image_dir, device, batch_size=16):
    """Extract CLIP ViT-B/16 image features."""
    try:
        import clip
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
        import clip

    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    dataset = ImageFolderDataset(image_dir, transform=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().float())
            all_paths.extend(paths)

    return {
        "paths": all_paths,
        "features": torch.cat(all_features, dim=0),  # [N, 512]
    }


def process_scene(dataset_name, scene, model_name, device, batch_size=8):
    """Extract features for a single scene."""
    cfg = DATASET_CONFIGS[dataset_name]
    image_dir = os.path.join(cfg["data_dir"], scene, cfg["image_subdir"])

    if not os.path.isdir(image_dir):
        # Try without subdirectory
        image_dir = os.path.join(cfg["data_dir"], scene, "images")
        if not os.path.isdir(image_dir):
            print(f"  Image directory not found for {scene}")
            return

    out_dir = os.path.join(FEATURES_BASE, dataset_name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene}.pt")

    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    print(f"  Extracting {model_name} features from {image_dir}")

    if model_name == "dinov2":
        result = extract_dinov2(image_dir, device, batch_size=batch_size)
    elif model_name == "alexnet":
        result = extract_alexnet(image_dir, device, batch_size=batch_size)
    elif model_name == "clip":
        result = extract_clip(image_dir, device, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    torch.save(result, out_path)
    n_images = len(result["paths"])
    print(f"  Saved {n_images} image features to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and cache visual features")
    parser.add_argument("--dataset", type=str, required=True, choices=["llff", "tt", "ns", "all"])
    parser.add_argument("--model", type=str, default="all", choices=["dinov2", "alexnet", "clip", "all"])
    parser.add_argument("--scene", type=str, help="Process single scene")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="Overwrite existing features")
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    models = ["dinov2", "alexnet", "clip"] if args.model == "all" else [args.model]

    for ds in datasets:
        cfg = DATASET_CONFIGS[ds]
        scenes = [args.scene] if args.scene else cfg["scenes"]

        for model_name in models:
            print(f"\n=== {ds} / {model_name} ===")
            for scene in scenes:
                try:
                    process_scene(ds, scene, model_name, args.device, args.batch_size)
                except Exception as e:
                    print(f"  ERROR {scene}: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()

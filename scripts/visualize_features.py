import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os, sys
from sklearn.decomposition import PCA

def load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model.eval().cuda()
    return model

def extract_features(model, img_path, layer=4):
    img = Image.open(img_path).convert("RGB")
    # Resize to 518x518
    img = img.resize((518, 518))
    x = torch.from_numpy(np.array(img)).float().permute(2,0,1).unsqueeze(0) / 255.0
    x = x.cuda()
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
    x = (x - mean) / std
    
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    handles = []
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(hook_fn(f"block_{i}")))
    
    with torch.no_grad():
        model(x)
    
    for h in handles:
        h.remove()
    
    feat = features[f"block_{layer}"]  # (1, 1370, 768)
    # Remove CLS token, reshape to spatial grid
    feat = feat[:, 1:, :]  # (1, 1369, 768)
    h = w = 37
    feat = feat.reshape(1, h, w, 768)
    return feat.cpu().numpy()[0], img

def visualize_pca(features, n_components=3):
    """PCA on features, map first 3 components to RGB."""
    h, w, d = features.shape
    flat = features.reshape(-1, d)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(flat)  # (h*w, 3)
    # Normalize to [0, 1]
    proj = proj - proj.min(axis=0)
    proj = proj / (proj.max(axis=0) + 1e-8)
    rgb = (proj * 255).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(rgb).resize((518, 518), Image.NEAREST)

def visualize_norm(features):
    """Feature norm heatmap."""
    norm = np.linalg.norm(features, axis=-1)  # (h, w)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    # Apply colormap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    colored = cm.viridis(norm)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored).resize((518, 518), Image.NEAREST)

if __name__ == "__main__":
    data_dir = sys.argv[1]  # e.g., /gpfs/workdir/malhotraa/data/LLFF/horns
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick a few images
    img_dir = os.path.join(data_dir, "images_4")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_dir, "images")
    
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(("jpg", "png", "JPG"))])
    # Pick 4 evenly spaced images
    indices = [0, len(imgs)//4, len(imgs)//2, 3*len(imgs)//4]
    
    model = load_dinov2()
    
    for layer in [4, 8]:
        for idx in indices:
            img_path = os.path.join(img_dir, imgs[idx])
            print(f"Processing {imgs[idx]} layer {layer}...")
            feat, orig_img = extract_features(model, img_path, layer=layer)
            
            pca_img = visualize_pca(feat)
            norm_img = visualize_norm(feat)
            
            # Save
            tag = os.path.splitext(imgs[idx])[0]
            orig_resized = orig_img.resize((518, 518))
            orig_resized.save(os.path.join(output_dir, f"{tag}_orig.png"))
            pca_img.save(os.path.join(output_dir, f"{tag}_L{layer}_pca.png"))
            norm_img.save(os.path.join(output_dir, f"{tag}_L{layer}_norm.png"))
    
    print("Done!")

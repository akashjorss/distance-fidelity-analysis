#!/usr/bin/env python3
"""Compute PSNR, SSIM, LPIPS for NeRF test renders vs ground truth.

Usage:
    python compute_nerf_metrics.py
"""
import os
import json
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

WORKDIR = '/gpfs/workdir/malhotraa'
RESULTS = f'{WORKDIR}/ConMax3D_reproduce/results'

SCENES_CONFIG = {
    'fern':     ('LLFF', 4, 0),
    'flower':   ('LLFF', 4, 0),
    'fortress': ('LLFF', 4, 0),
    'horns':    ('LLFF', 4, 0),
    'leaves':   ('LLFF', 4, 0),
    'orchids':  ('LLFF', 4, 0),
    'room':     ('LLFF', 4, 0),
    'trex':     ('LLFF', 4, 0),
    'Ballroom': ('Tanks', 1, 0),
    'Barn':     ('Tanks', 1, 0),
    'Church':   ('Tanks', 1, 150),
    'Family':   ('Tanks', 1, 150),
    'Francis':  ('Tanks', 1, 0),
    'Horse':    ('Tanks', 1, 0),
    'Ignatius': ('Tanks', 1, 0),
    'Museum':   ('Tanks', 1, 0),
}

# method_name -> (nerf_result_dir, suffix_in_dir, selection_dir, strategy_name)
METHODS = {
    'infomax':  ('v3_nerf_infomax',      'infomax',      'v3_dinov2_L4',    'infomax'),
    'fvs':      ('v3_nerf_fvs',          'fvs',          'v3',              'fvs'),
    'random':   ('v3_nerf_random_s42',   'random_s42',   'v3_random_s42',   'random'),
    'plucker':  ('v3_nerf_fvs_plucker',  'fvs_plucker',  'v3_fvs_plucker',  'fvs_plucker'),
}

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img).astype(np.float32) / 255.0

def get_gt_images(scene, dataset, data_factor, max_images):
    data_base = f'{WORKDIR}/data/{dataset}'
    if data_factor > 1:
        imgdir = os.path.join(data_base, scene, f'images_{data_factor}')
    else:
        imgdir = os.path.join(data_base, scene, 'images')

    if not os.path.isdir(imgdir):
        return [], imgdir

    imgs = sorted([f for f in os.listdir(imgdir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if max_images > 0:
        imgs = imgs[:max_images]
    return [os.path.join(imgdir, f) for f in imgs], imgdir

def get_test_indices(all_image_paths, selection_dir, scene, strategy):
    sel_file = os.path.join(RESULTS, selection_dir, scene,
                            f'train_indices_{scene}_{strategy}.json')
    if not os.path.exists(sel_file):
        return None, sel_file
    with open(sel_file) as f:
        d = json.load(f)
    train_idx = set(d['selected_indices'])
    n = len(all_image_paths)
    test_idx = sorted([i for i in range(n) if i not in train_idx])
    return test_idx, sel_file

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    all_results = {}

    for method_name, (nerf_dir, suffix, sel_dir, strategy) in METHODS.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")
        method_results = {}

        for scene, (dataset, data_factor, max_images) in SCENES_CONFIG.items():
            render_dir = os.path.join(RESULTS, nerf_dir, scene,
                                      f'{scene}_{suffix}', 'testset_050000')

            if not os.path.isdir(render_dir):
                print(f"  {scene}: no renders")
                continue

            all_imgs, imgdir = get_gt_images(scene, dataset, data_factor, max_images)
            if not all_imgs:
                print(f"  {scene}: no GT images at {imgdir}")
                continue

            test_indices, sel_file = get_test_indices(all_imgs, sel_dir, scene, strategy)
            if test_indices is None:
                print(f"  {scene}: no selection file at {sel_file}")
                continue

            render_files = sorted([f for f in os.listdir(render_dir)
                                   if f.endswith('.png') and f[0].isdigit()])
            n_renders = len(render_files)

            if len(test_indices) != n_renders:
                print(f"  {scene}: WARNING {n_renders} renders vs {len(test_indices)} test imgs")
                n = min(n_renders, len(test_indices))
                test_indices = test_indices[:n]

            psnrs, ssims, lpipses = [], [], []

            for i, tidx in enumerate(test_indices):
                rpath = os.path.join(render_dir, f'{i:03d}.png')
                gpath = all_imgs[tidx]

                if not os.path.exists(rpath):
                    continue

                render_img = load_image(rpath)
                gt_img = load_image(gpath)

                # Resize GT if needed
                if render_img.shape != gt_img.shape:
                    gt_pil = Image.open(gpath).convert('RGB')
                    gt_pil = gt_pil.resize((render_img.shape[1], render_img.shape[0]), Image.LANCZOS)
                    gt_img = np.array(gt_pil).astype(np.float32) / 255.0

                # PSNR
                mse = np.mean((render_img - gt_img) ** 2)
                psnr = -10.0 * np.log10(mse) if mse > 0 else 100.0
                psnrs.append(psnr)

                # SSIM
                s = ssim(render_img, gt_img, channel_axis=2, data_range=1.0)
                ssims.append(s)

                # LPIPS
                r_t = torch.from_numpy(render_img).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                g_t = torch.from_numpy(gt_img).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
                with torch.no_grad():
                    lp = lpips_fn(r_t, g_t).item()
                lpipses.append(lp)

            if not psnrs:
                print(f"  {scene}: no valid pairs")
                continue

            mean_psnr = float(np.mean(psnrs))
            mean_ssim = float(np.mean(ssims))
            mean_lpips = float(np.mean(lpipses))

            # Save metrics
            metrics = {
                'psnr': mean_psnr,
                'ssim': mean_ssim,
                'lpips': mean_lpips,
                'n_test': len(psnrs),
                'iter': 50000
            }

            metrics_file = os.path.join(render_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            method_results[scene] = metrics
            print(f"  {scene}: PSNR={mean_psnr:.4f} SSIM={mean_ssim:.4f} LPIPS={mean_lpips:.4f} ({len(psnrs)} test)")

        # Print averages
        llff_scenes = [s for s in ['fern','flower','fortress','horns','leaves','orchids','room','trex'] if s in method_results]
        tt_scenes = [s for s in ['Ballroom','Barn','Church','Family','Francis','Horse','Ignatius','Museum'] if s in method_results]

        for label, scene_list in [('LLFF', llff_scenes), ('T&T', tt_scenes)]:
            if scene_list:
                avg_p = np.mean([method_results[s]['psnr'] for s in scene_list])
                avg_s = np.mean([method_results[s]['ssim'] for s in scene_list])
                avg_l = np.mean([method_results[s]['lpips'] for s in scene_list])
                print(f"  {label} avg: PSNR={avg_p:.4f} SSIM={avg_s:.4f} LPIPS={avg_l:.4f} ({len(scene_list)} scenes)")

        all_results[method_name] = method_results

    # Save summary
    summary_file = os.path.join(RESULTS, 'nerf_metrics_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

if __name__ == '__main__':
    main()

"""Preprocess NeRF Synthetic datasets to gsplat-compatible format.

For each scene:
1. Read transforms_train.json (camera_angle_x + frames)
2. Compute fl_x = fl_y = 0.5 * W / tan(camera_angle_x / 2) where W=800
3. Add w, h, fl_x, fl_y fields to transforms.json
4. Fix file_path: append .png, point to images/ dir
5. Composite RGBA -> RGB on white background, save to images/
6. Write transforms.json alongside images/

Usage:
    python prep_nerf_synthetic.py --data_dir /path/to/nerf_synthetic --output_dir /path/to/output
    python prep_nerf_synthetic.py --data_dir /path/to/nerf_synthetic/lego --scene lego --output_dir /path/to/output
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from PIL import Image
from tqdm import tqdm


SCENES = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
IMG_SIZE = 800  # NeRF Synthetic images are 800x800


def composite_rgba_to_rgb(rgba_path, output_path):
    """Composite RGBA image onto white background and save as RGB PNG."""
    img = Image.open(rgba_path).convert('RGBA')
    # Create white background
    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    # Composite
    composited = Image.alpha_composite(bg, img)
    # Convert to RGB and save
    composited.convert('RGB').save(output_path)


def process_scene(scene_dir, output_dir, split='train'):
    """Process a single NeRF Synthetic scene."""
    scene_name = os.path.basename(scene_dir)
    transforms_path = os.path.join(scene_dir, f'transforms_{split}.json')

    if not os.path.exists(transforms_path):
        print(f"WARNING: {transforms_path} not found, skipping")
        return

    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    camera_angle_x = meta['camera_angle_x']
    W = IMG_SIZE
    H = IMG_SIZE

    # Compute focal length
    fl_x = 0.5 * W / math.tan(camera_angle_x / 2.0)
    fl_y = fl_x  # Square pixels

    # Create output directories
    scene_output = os.path.join(output_dir, scene_name)
    images_output = os.path.join(scene_output, 'images')
    os.makedirs(images_output, exist_ok=True)

    # Process frames
    new_frames = []
    for i, frame in enumerate(tqdm(meta['frames'], desc=f'{scene_name}/{split}')):
        old_path = frame['file_path']
        # Original paths are like ./train/r_0 (no .png extension)
        # Build the actual source path
        rel_path = old_path.lstrip('./')
        src_path = os.path.join(scene_dir, rel_path + '.png')

        if not os.path.exists(src_path):
            print(f"WARNING: {src_path} not found, skipping frame {i}")
            continue

        # Output filename: r_0.png, r_1.png, etc.
        img_name = os.path.basename(rel_path) + '.png'
        dst_path = os.path.join(images_output, img_name)

        # Composite RGBA -> RGB on white background
        composite_rgba_to_rgb(src_path, dst_path)

        # Build new frame entry
        new_frame = {
            'file_path': f'images/{img_name}',
            'transform_matrix': frame['transform_matrix'],
        }
        new_frames.append(new_frame)

    # Write transforms.json
    transforms_out = {
        'w': W,
        'h': H,
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': W / 2.0,
        'cy': H / 2.0,
        'frames': new_frames,
    }

    out_path = os.path.join(scene_output, 'transforms.json')
    with open(out_path, 'w') as f:
        json.dump(transforms_out, f, indent=2)

    print(f'{scene_name}: {len(new_frames)} frames -> {out_path}')
    return len(new_frames)


def process_test_split(scene_dir, output_dir):
    """Process test split and write transforms_test.json for evaluation."""
    scene_name = os.path.basename(scene_dir)
    transforms_path = os.path.join(scene_dir, 'transforms_test.json')

    if not os.path.exists(transforms_path):
        print(f"WARNING: {transforms_path} not found, skipping test split")
        return

    with open(transforms_path, 'r') as f:
        meta = json.load(f)

    camera_angle_x = meta['camera_angle_x']
    W = IMG_SIZE
    H = IMG_SIZE
    fl_x = 0.5 * W / math.tan(camera_angle_x / 2.0)
    fl_y = fl_x

    scene_output = os.path.join(output_dir, scene_name)
    test_images_output = os.path.join(scene_output, 'test_images')
    os.makedirs(test_images_output, exist_ok=True)

    new_frames = []
    for i, frame in enumerate(tqdm(meta['frames'], desc=f'{scene_name}/test')):
        old_path = frame['file_path']
        rel_path = old_path.lstrip('./')
        src_path = os.path.join(scene_dir, rel_path + '.png')

        if not os.path.exists(src_path):
            continue

        img_name = os.path.basename(rel_path) + '.png'
        dst_path = os.path.join(test_images_output, img_name)
        composite_rgba_to_rgb(src_path, dst_path)

        new_frame = {
            'file_path': f'test_images/{img_name}',
            'transform_matrix': frame['transform_matrix'],
        }
        new_frames.append(new_frame)

    transforms_out = {
        'w': W,
        'h': H,
        'fl_x': fl_x,
        'fl_y': fl_y,
        'cx': W / 2.0,
        'cy': H / 2.0,
        'frames': new_frames,
    }

    out_path = os.path.join(scene_output, 'transforms_test.json')
    with open(out_path, 'w') as f:
        json.dump(transforms_out, f, indent=2)

    print(f'{scene_name} test: {len(new_frames)} frames -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess NeRF Synthetic for gsplat')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to nerf_synthetic/ directory (or single scene dir)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--scene', type=str, default=None,
                        help='Single scene to process (default: all)')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test split processing')
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = SCENES

    for scene in scenes:
        scene_dir = os.path.join(args.data_dir, scene) if args.scene is None else args.data_dir
        if not os.path.isdir(scene_dir):
            # Try as subdirectory
            scene_dir = os.path.join(args.data_dir, scene)
        if not os.path.isdir(scene_dir):
            print(f"WARNING: Scene dir not found: {scene_dir}, skipping")
            continue

        process_scene(scene_dir, args.output_dir, split='train')
        if not args.skip_test:
            process_test_split(scene_dir, args.output_dir)

    print("\nDone! Preprocessed data written to:", args.output_dir)


if __name__ == '__main__':
    main()

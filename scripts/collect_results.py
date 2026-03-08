"""Collect and aggregate results across all datasets and methods.

Walks result directories, extracts PSNR/SSIM/LPIPS + timing,
produces per-scene tables, dataset averages, CSV, and LaTeX output.

Usage:
    python collect_results.py --results_dir /path/to/results --output_dir /path/to/output
"""

import os
import json
import argparse
import csv
from collections import defaultdict


# ActiveNeRF reference numbers (Table 1, Setting II, 10 images, NeRF Synthetic)
ACTIVENERF_REFERENCE = {
    'nerf_rand': 28.04,
    'activenerf': 28.51,
}

LLFF_SCENES = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
TT_SCENES = ['Ballroom', 'Barn', 'Church', 'Family', 'Francis', 'Horse', 'Ignatius', 'Museum']
NS_SCENES = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
METHODS = ['conmax3d', 'random', 'fvs']
BUDGETS = ['ten', 'twenty', 'twenty_five']


def find_best_stats_file(stats_dir):
    """Find the stats file with the highest step number."""
    if not os.path.isdir(stats_dir):
        return None
    stats_files = [f for f in os.listdir(stats_dir) if f.startswith('val_step') and f.endswith('.json')]
    if not stats_files:
        return None
    # Sort by step number
    stats_files.sort(key=lambda x: int(x.replace('val_step', '').replace('.json', '')))
    return os.path.join(stats_dir, stats_files[-1])


def load_gsplat_stats(stats_path):
    """Load metrics from a gsplat stats JSON file."""
    with open(stats_path, 'r') as f:
        data = json.load(f)
    return {
        'psnr': data.get('psnr', None),
        'ssim': data.get('ssim', None),
        'lpips': data.get('lpips', None),
        'num_gaussians': data.get('num_GS', data.get('num_gaussians', None)),
    }


def load_nerf_stats(result_dir, scene, method):
    """Load metrics from vanilla NeRF test set evaluation."""
    expname = f'{scene}_{method}'
    testset_dir = os.path.join(result_dir, expname)

    if not os.path.isdir(testset_dir):
        return None

    # Look for testset_XXXXXX directories
    testset_dirs = [d for d in os.listdir(testset_dir)
                    if d.startswith('testset_') and os.path.isdir(os.path.join(testset_dir, d))]
    if not testset_dirs:
        return None

    # Use the latest testset
    testset_dirs.sort(key=lambda x: int(x.split('_')[1]))
    latest = os.path.join(testset_dir, testset_dirs[-1])

    # Check for metrics file
    metrics_file = os.path.join(latest, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return json.load(f)

    # Try to compute PSNR from rendered images if available
    return None


def load_timing(results_base, dataset_prefix, scene):
    """Load ConMax3D selection timing for a scene."""
    time_file = os.path.join(results_base, dataset_prefix, 'time', f'execution_time_{scene}.json')
    if os.path.exists(time_file):
        with open(time_file) as f:
            data = json.load(f)
        return sum(data.values())
    return None


def collect_gsplat_results(results_base, gsplat_dir, scenes, dataset_name):
    """Collect 3DGS results for a dataset."""
    results = []

    for budget in BUDGETS:
        for method in METHODS:
            for scene in scenes:
                stats_dir = os.path.join(results_base, gsplat_dir, budget, method, scene, 'stats')
                stats_file = find_best_stats_file(stats_dir)

                if stats_file:
                    metrics = load_gsplat_stats(stats_file)
                    results.append({
                        'dataset': dataset_name,
                        'scene': scene,
                        'method': method,
                        'budget': budget,
                        'model': '3DGS',
                        **metrics,
                    })
                else:
                    results.append({
                        'dataset': dataset_name,
                        'scene': scene,
                        'method': method,
                        'budget': budget,
                        'model': '3DGS',
                        'psnr': None,
                        'ssim': None,
                        'lpips': None,
                        'num_gaussians': None,
                    })

    return results


def collect_nerf_results(results_base, scenes):
    """Collect vanilla NeRF results for NeRF Synthetic."""
    results = []
    nerf_dir = os.path.join(results_base, 'nerf_ns', 'ten')

    for method in METHODS:
        for scene in scenes:
            scene_dir = os.path.join(nerf_dir, method, scene)
            metrics = load_nerf_stats(scene_dir, scene, method)

            results.append({
                'dataset': 'nerf_synthetic',
                'scene': scene,
                'method': method,
                'budget': 'ten',
                'model': 'NeRF',
                'psnr': metrics.get('psnr') if metrics else None,
                'ssim': metrics.get('ssim') if metrics else None,
                'lpips': metrics.get('lpips') if metrics else None,
            })

    return results


def compute_averages(results):
    """Compute per-dataset per-method averages."""
    grouped = defaultdict(list)
    for r in results:
        key = (r['dataset'], r['method'], r['budget'], r['model'])
        if r['psnr'] is not None:
            grouped[key].append(r)

    averages = []
    for key, entries in grouped.items():
        dataset, method, budget, model = key
        avg = {
            'dataset': dataset,
            'method': method,
            'budget': budget,
            'model': model,
            'scene': 'AVERAGE',
            'psnr': sum(e['psnr'] for e in entries) / len(entries),
            'ssim': sum(e['ssim'] for e in entries if e.get('ssim') is not None) / max(1, sum(1 for e in entries if e.get('ssim') is not None)),
            'lpips': sum(e['lpips'] for e in entries if e.get('lpips') is not None) / max(1, sum(1 for e in entries if e.get('lpips') is not None)),
            'n_scenes': len(entries),
        }
        averages.append(avg)

    return averages


def write_csv(results, output_path):
    """Write results to CSV."""
    if not results:
        return

    fieldnames = ['dataset', 'scene', 'method', 'budget', 'model', 'psnr', 'ssim', 'lpips', 'num_gaussians']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV written to: {output_path}")


def format_metric(val, fmt='.2f'):
    return f'{val:{fmt}}' if val is not None else '-'


def write_latex_table(results, averages, dataset_name, budget, output_path, model='3DGS'):
    """Write LaTeX comparison table for a specific dataset and budget."""
    # Filter results
    filtered = [r for r in results if r['dataset'] == dataset_name and r['budget'] == budget and r['model'] == model]
    avg_filtered = [a for a in averages if a['dataset'] == dataset_name and a['budget'] == budget and a['model'] == model]

    if not filtered:
        return

    # Get scenes
    scenes = sorted(set(r['scene'] for r in filtered if r['scene'] != 'AVERAGE'))

    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + f'{dataset_name.upper()} {model} results (k={budget})' + r'}')
    cols = 'l' + 'c' * len(scenes) + 'c'
    lines.append(r'\begin{tabular}{' + cols + r'}')
    lines.append(r'\toprule')
    header = 'Method & ' + ' & '.join(scenes) + r' & Avg \\'
    lines.append(header)
    lines.append(r'\midrule')

    for method in METHODS:
        method_results = {r['scene']: r for r in filtered if r['method'] == method}
        method_avg = next((a for a in avg_filtered if a['method'] == method), None)

        row_vals = []
        for scene in scenes:
            r = method_results.get(scene)
            row_vals.append(format_metric(r['psnr'] if r else None))

        avg_val = format_metric(method_avg['psnr'] if method_avg else None)
        method_display = {'conmax3d': r'\textbf{ConMax3D}', 'random': 'Random', 'fvs': 'FVS'}.get(method, method)
        row = f'{method_display} & ' + ' & '.join(row_vals) + f' & {avg_val}' + r' \\'
        lines.append(row)

    # Add ActiveNeRF reference if NeRF Synthetic + NeRF model + k=10
    if dataset_name == 'nerf_synthetic' and model == 'NeRF' and budget == 'ten':
        lines.append(r'\midrule')
        lines.append(r'NeRF+Rand (ActiveNeRF) & \multicolumn{' + str(len(scenes)) + r'}{c}{-} & ' + f'{ACTIVENERF_REFERENCE["nerf_rand"]:.2f}' + r' \\')
        lines.append(r'ActiveNeRF & \multicolumn{' + str(len(scenes)) + r'}{c}{-} & ' + f'{ACTIVENERF_REFERENCE["activenerf"]:.2f}' + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table written to: {output_path}")


def print_summary(results, averages):
    """Print summary to stdout."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for dataset in ['llff', 'tt', 'nerf_synthetic']:
        for model in ['3DGS', 'NeRF']:
            for budget in BUDGETS:
                budget_avg = [a for a in averages
                             if a['dataset'] == dataset and a['budget'] == budget and a['model'] == model]
                if not budget_avg:
                    continue

                budget_display = {'ten': 'k=10', 'twenty': 'k=20', 'twenty_five': 'k=25'}[budget]
                print(f"\n{dataset.upper()} ({model}, {budget_display}):")
                for method in METHODS:
                    avg = next((a for a in budget_avg if a['method'] == method), None)
                    if avg:
                        psnr = format_metric(avg['psnr'])
                        ssim = format_metric(avg['ssim'], '.3f')
                        lpips = format_metric(avg['lpips'], '.3f')
                        print(f"  {method:12s}: PSNR={psnr}  SSIM={ssim}  LPIPS={lpips}  ({avg.get('n_scenes', '?')} scenes)")


def main():
    parser = argparse.ArgumentParser(description='Collect and aggregate ConMax3D results')
    parser.add_argument('--results_dir', type=str,
                        default='/gpfs/workdir/malhotraa/ConMax3D_reproduce/results',
                        help='Base results directory')
    parser.add_argument('--output_dir', type=str,
                        default='/gpfs/workdir/malhotraa/ConMax3D_reproduce/results/tables',
                        help='Output directory for tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    # Collect 3DGS results
    print("Collecting LLFF 3DGS results...")
    all_results.extend(collect_gsplat_results(args.results_dir, 'gsplat', LLFF_SCENES, 'llff'))

    print("Collecting T&T 3DGS results...")
    all_results.extend(collect_gsplat_results(args.results_dir, 'gsplat_tt', TT_SCENES, 'tt'))

    print("Collecting NeRF Synthetic 3DGS results...")
    all_results.extend(collect_gsplat_results(args.results_dir, 'gsplat_ns', NS_SCENES, 'nerf_synthetic'))

    # Collect vanilla NeRF results
    print("Collecting NeRF Synthetic vanilla NeRF results...")
    all_results.extend(collect_nerf_results(args.results_dir, NS_SCENES))

    # Compute averages
    averages = compute_averages(all_results)

    # Write CSV
    write_csv(all_results, os.path.join(args.output_dir, 'all_results.csv'))
    write_csv(averages, os.path.join(args.output_dir, 'averages.csv'))

    # Write LaTeX tables
    for dataset, scenes_list in [('llff', LLFF_SCENES), ('tt', TT_SCENES), ('nerf_synthetic', NS_SCENES)]:
        for budget in BUDGETS:
            latex_path = os.path.join(args.output_dir, f'{dataset}_{budget}_3dgs.tex')
            write_latex_table(all_results, averages, dataset, budget, latex_path, model='3DGS')

    # NeRF Synthetic vanilla NeRF table
    latex_path = os.path.join(args.output_dir, 'nerf_synthetic_ten_nerf.tex')
    write_latex_table(all_results, averages, 'nerf_synthetic', 'ten', latex_path, model='NeRF')

    # Print summary
    print_summary(all_results, averages)


if __name__ == '__main__':
    main()

#!/bin/bash
#SBATCH --job-name=v3_swap
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/swap_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/swap_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# Gsplat training on swap-improved InfoMax3D indices (LLFF only)
# Submit:
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-7 slurm/test_swap_gsplat.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

ENTROPY_DIR=$WORKDIR/ConMax3D_reproduce/results/entropy_analysis
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_swap

SCENES=(fern flower fortress horns leaves orchids room trex)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}

echo "=== [$IDX] $SCENE | Swap gsplat ==="

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

INDICES_FILE="$ENTROPY_DIR/train_indices_${SCENE}_infomax_swap.json"
if [ ! -f "$INDICES_FILE" ]; then
    echo "SKIP: no swap indices file"
    exit 1
fi

INDICES=$($GSPLAT_PYTHON -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
" "$INDICES_FILE")

echo "=== Indices: $INDICES ==="

GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_infomax_swap

$GSPLAT_PYTHON $GSPLAT_TRAINER default \
    --data_dir $WORKDIR/data/LLFF/$SCENE \
    --dataset_type colmap \
    --data_factor 4 \
    --init_type sfm \
    --train_indices "$INDICES" \
    --result_dir $GSPLAT_RESULT \
    --max_steps 30000 \
    --eval_steps 30000 \
    --save_steps 30000 \
    --disable_viewer

echo "=== RESULT ==="
$GSPLAT_PYTHON -c "
import json, os, sys
stats_dir = sys.argv[1]
if os.path.isdir(stats_dir):
    for f in sorted(os.listdir(stats_dir)):
        if f.endswith('.json') and f.startswith('val'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            psnr = s.get('psnr', 0)
            ssim = s.get('ssim', 0)
            lpips_val = s.get('lpips', 0)
            print('%s/swap: PSNR=%.4f SSIM=%.4f LPIPS=%.4f' % ('$SCENE', psnr, ssim, lpips_val))
" "$GSPLAT_RESULT/stats"

echo "=== Done: $SCENE/swap ==="

#!/bin/bash
#SBATCH --job-name=v3_pfvs
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3p_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3p_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# Pose-based FVS baselines: euclidean, angular, plucker
# Submit:
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=FVS_MODE=euclidean slurm/test_v3_pose_fvs.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=FVS_MODE=angular slurm/test_v3_pose_fvs.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=FVS_MODE=plucker slurm/test_v3_pose_fvs.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

FVS_MODE=${FVS_MODE:-euclidean}
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_fvs_${FVS_MODE}

SCENES=(fern flower fortress horns leaves orchids room trex \
        Ballroom Barn Church Family Francis Horse Ignatius Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
DATA_FACTORS=(4 4 4 4 4 4 4 4 1 1 1 1 1 1 1 1)
MAX_IMAGES=(0 0 0 0 0 0 0 0 0 0 150 150 0 0 0 0)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}
MAXIMG=${MAX_IMAGES[$IDX]}

K=10

echo "=== [$IDX] $SCENE | FVS $FVS_MODE ==="

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# Phase 1: Pose-based FVS selection
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

MAX_IMAGES_FLAG=""
if [ "$MAXIMG" -gt 0 ]; then
    MAX_IMAGES_FLAG="--max_images $MAXIMG"
fi

python $SCRIPTS/infomax3d.py \
    --data_dir $DATA_BASE \
    --scene $SCENE \
    --output_dir $RESULTS/$SCENE \
    --k $K \
    --data_factor $DFACTOR \
    --strategies fvs_${FVS_MODE} \
    --batch_size 8 \
    $MAX_IMAGES_FLAG

# Phase 2: gsplat
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

STRAT="fvs_${FVS_MODE}"
INDICES_FILE="$RESULTS/$SCENE/train_indices_${SCENE}_${STRAT}.json"
if [ ! -f "$INDICES_FILE" ]; then
    echo "SKIP: no indices file"
    exit 1
fi

INDICES=$($GSPLAT_PYTHON -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
" "$INDICES_FILE")

echo "=== gsplat: $SCENE/$STRAT | Indices: $INDICES ==="

GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_${STRAT}

$GSPLAT_PYTHON $GSPLAT_TRAINER default \
    --data_dir $DATA_BASE/$SCENE \
    --dataset_type colmap \
    --data_factor $DFACTOR \
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
        if f.endswith('.json'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            psnr = s.get('psnr', 0)
            ssim = s.get('ssim', 0)
            lpips_val = s.get('lpips', 0)
            print(f'${SCENE}/${STRAT}: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips_val:.4f}')
" "$GSPLAT_RESULT/stats"

echo "=== Done: $SCENE/$STRAT ==="

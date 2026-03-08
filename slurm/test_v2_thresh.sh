#!/bin/bash
#SBATCH --job-name=v2_thresh
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2t_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2t_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --exclude=ruche-gpu18
#SBATCH --export=NONE

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v2_thresh
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

# 8 configs: 4 scenes × 2 thresholds
SCENES=(fern trex Church Museum fern trex Church Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/Tanks $WORKDIR/data/Tanks \
            $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
DATA_FACTORS=(4 4 1 1 4 4 1 1)
MAX_IMAGES=(0 0 150 0 0 0 150 0)
THRESHOLDS=(5 5 5 5 40 40 40 40)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}
MAXIMG=${MAX_IMAGES[$IDX]}
THRESH=${THRESHOLDS[$IDX]}

K=10
TAG="${SCENE}_t${THRESH}"

echo "=== $TAG: threshold=$THRESH ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# ── Phase 1: SAM2 + strategies ──
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKDIR/segment-anything-2:$PYTHONPATH

MAX_IMAGES_FLAG=""
if [ "$MAXIMG" -gt 0 ]; then
    MAX_IMAGES_FLAG="--max_images $MAXIMG"
fi

python $SCRIPTS/conmax3d_v2.py \
    --data_dir $DATA_BASE \
    --scene $SCENE \
    --output_dir $RESULTS/$TAG \
    --k $K \
    --data_factor $DFACTOR \
    --n_concepts $((K * 2)) \
    --epipolar_threshold $THRESH \
    --strategies concept_filtered_fvs,pure_fvs \
    $MAX_IMAGES_FLAG

# ── Phase 2: gsplat for concept_filtered_fvs only ──
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

STRAT="concept_filtered_fvs"
INDICES_FILE="$RESULTS/$TAG/train_indices_${SCENE}_${STRAT}.json"

INDICES=$(python -c "
import json
with open('$INDICES_FILE') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")
echo "=== gsplat: $TAG/$STRAT | Indices: $INDICES ==="

GSPLAT_RESULT=$RESULTS/$TAG/gsplat_$STRAT

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

echo "=== RESULT: $TAG/$STRAT ==="
python -c "
import json, os
stats_dir = '$GSPLAT_RESULT/stats'
if os.path.isdir(stats_dir):
    for f in sorted(os.listdir(stats_dir)):
        if f.endswith('.json'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            psnr = s.get('psnr', 0)
            ssim = s.get('ssim', 0)
            lpips = s.get('lpips', 0)
            print(f'$TAG/$STRAT: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}')
"

echo "=== Done: $TAG ==="

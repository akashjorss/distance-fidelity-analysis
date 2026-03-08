#!/bin/bash
#SBATCH --job-name=v3_sweep
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3s_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3s_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# InfoMax3D v3 sweep: different layers and backbones
# Usage:
#   BACKBONE=dinov2 LAYER=4 sbatch --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_sweep.sh
#   BACKBONE=dinov2 LAYER=6 sbatch --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_sweep.sh
#   BACKBONE=dinov2 LAYER=10 sbatch --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_sweep.sh
#   BACKBONE=resnet LAYER=3 sbatch --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_sweep.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

# Use env vars for sweep params (with defaults)
BACKBONE=${BACKBONE:-dinov2}
LAYER=${LAYER:-8}
TAG="${BACKBONE}_L${LAYER}"

RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_${TAG}

# 16 scenes: 0-7 = LLFF, 8-15 = T&T
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
# Only run infomax for sweep (FVS doesn't change meaningfully across layers)
STRATEGIES="infomax"

echo "=== [$IDX] $SCENE | $TAG ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# ── Phase 1: Feature extraction + selection ──
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

MAX_IMAGES_FLAG=""
if [ "$MAXIMG" -gt 0 ]; then
    MAX_IMAGES_FLAG="--max_images $MAXIMG"
fi

BACKBONE_FLAGS=""
if [ "$BACKBONE" = "dinov2" ]; then
    BACKBONE_FLAGS="--backbone dinov2 --dino_layer $LAYER"
elif [ "$BACKBONE" = "resnet" ]; then
    BACKBONE_FLAGS="--backbone resnet --resnet_stage $LAYER"
fi

python $SCRIPTS/infomax3d.py \
    --data_dir $DATA_BASE \
    --scene $SCENE \
    --output_dir $RESULTS/$SCENE \
    --k $K \
    --data_factor $DFACTOR \
    --strategies $STRATEGIES \
    $BACKBONE_FLAGS \
    --batch_size 8 \
    $MAX_IMAGES_FLAG

# ── Phase 2: gsplat ──
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

for STRAT in infomax; do
    INDICES_FILE="$RESULTS/$SCENE/train_indices_${SCENE}_${STRAT}.json"
    if [ ! -f "$INDICES_FILE" ]; then
        echo "SKIP $STRAT: no indices file"
        continue
    fi

    INDICES=$(python -c "
import json
with open('$INDICES_FILE') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")
    echo ""
    echo "============================================"
    echo "=== gsplat: $SCENE/$TAG/$STRAT ==="
    echo "=== Indices: $INDICES ==="
    echo "============================================"

    GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_$STRAT

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

    echo "=== $SCENE/$TAG/$STRAT RESULT ==="
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
            print(f'$SCENE/$TAG/$STRAT: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}')
else:
    print('No stats found for $STRAT')
"
done

echo ""
echo "=== ALL DONE: $SCENE/$TAG ==="

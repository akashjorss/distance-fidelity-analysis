#!/bin/bash
#SBATCH --job-name=v3_multi
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3m_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3m_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# InfoMax3D v3 multi-layer experiments
# Usage (use _ separator for layers to avoid SLURM --export comma conflicts):
#   sbatch --export=NONE,BACKBONE=dinov2,LAYERS=4_6_8,MODE=concat --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_multilayer.sh
#   sbatch --export=NONE,BACKBONE=dinov2,LAYERS=4_6_8,MODE=sum --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_multilayer.sh
#   sbatch --export=NONE,BACKBONE=dinov2,LAYERS=4_6,MODE=concat --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_multilayer.sh
#   sbatch --export=NONE,BACKBONE=resnet,LAYERS=2_3_4,MODE=sum --partition=gpua100 --cpus-per-task=8 --array=0-15 slurm/test_v3_multilayer.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

BACKBONE=${BACKBONE:-dinov2}
LAYERS=${LAYERS:-4_6_8}
MODE=${MODE:-concat}
# Convert _ to , for Python args (avoid SLURM --export comma conflicts)
LAYERS_COMMA=$(echo "$LAYERS" | sed 's/_/,/g')
# Build tag from layers: e.g. dinov2_L4+6+8_concat or resnet_S2+3+4_sum
LAYER_TAG=$(echo "$LAYERS" | sed 's/_/+/g')
if [ "$BACKBONE" = "dinov2" ]; then
    TAG="${BACKBONE}_L${LAYER_TAG}_${MODE}"
    LAYER_FLAGS="--backbone dinov2 --dino_layers $LAYERS_COMMA --multilayer_mode $MODE"
else
    TAG="${BACKBONE}_S${LAYER_TAG}_${MODE}"
    LAYER_FLAGS="--backbone resnet --resnet_stages $LAYERS_COMMA --multilayer_mode $MODE"
fi

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
STRATEGIES="infomax"

echo "=== [$IDX] $SCENE | $TAG ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# ── Phase 1: Multi-layer feature extraction + selection ──
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
    --strategies $STRATEGIES \
    $LAYER_FLAGS \
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

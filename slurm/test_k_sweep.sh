#!/bin/bash
#SBATCH --job-name=k_sweep
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/ks_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/ks_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=ruche-gpu02,ruche-gpu07,ruche-gpu09,ruche-gpu10,ruche-gpu13,ruche-gpu18
#SBATCH --export=NONE
# K-sweep: run selection + gsplat for different k values and methods
# Submit (k=15,20,25 x 4 methods = 12 jobs):
#   for K in 15 20 25; do
#     sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=K=$K,METHOD=infomax slurm/test_k_sweep.sh
#     sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=K=$K,METHOD=fvs slurm/test_k_sweep.sh
#     sbatch --partition=gpua100 --cpus-per-task=8 --array=0-15 --export=K=$K,METHOD=fvs_plucker slurm/test_k_sweep.sh
#     sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=K=$K,METHOD=random slurm/test_k_sweep.sh
#   done

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

K=${K:-20}
METHOD=${METHOD:-infomax}

RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_k${K}_${METHOD}

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

echo "=== [$IDX] $SCENE | K=$K METHOD=$METHOD ==="

# Count available images to check if k < N (need at least 1 test image)
if [ "$DFACTOR" -gt 1 ]; then
    IMGDIR=$DATA_BASE/$SCENE/images_${DFACTOR}
else
    IMGDIR=$DATA_BASE/$SCENE/images
fi
if [ "$MAXIMG" -gt 0 ]; then
    N_IMAGES=$MAXIMG
else
    N_IMAGES=$(ls "$IMGDIR" 2>/dev/null | wc -l)
fi
if [ "$K" -ge "$N_IMAGES" ]; then
    echo "SKIP: K=$K >= N=$N_IMAGES for $SCENE, no test images possible"
    exit 0
fi

# Phase 1: Selection (needs seva env for DINOv2)
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

mkdir -p $RESULTS/$SCENE

MAX_IMAGES_FLAG=""
if [ "$MAXIMG" -gt 0 ]; then
    MAX_IMAGES_FLAG="--max_images $MAXIMG"
fi

# Map METHOD to strategy name for infomax3d.py
case $METHOD in
    infomax)
        STRATEGY="infomax"
        DINO_LAYER="--dino_layers 4"
        ;;
    fvs)
        STRATEGY="fvs"
        DINO_LAYER=""
        ;;
    fvs_plucker)
        STRATEGY="fvs_plucker"
        DINO_LAYER=""
        ;;
    random)
        STRATEGY="random"
        DINO_LAYER=""
        ;;
    *)
        echo "Unknown METHOD: $METHOD"
        exit 1
        ;;
esac

python $SCRIPTS/infomax3d.py \
    --data_dir $DATA_BASE \
    --scene $SCENE \
    --output_dir $RESULTS/$SCENE \
    --k $K \
    --data_factor $DFACTOR \
    --strategies $STRATEGY \
    --batch_size 8 \
    $DINO_LAYER \
    $MAX_IMAGES_FLAG

# Phase 2: gsplat training
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

INDICES_FILE="$RESULTS/$SCENE/train_indices_${SCENE}_${STRATEGY}.json"
if [ ! -f "$INDICES_FILE" ]; then
    echo "SKIP: no indices file at $INDICES_FILE"
    exit 1
fi

INDICES=$($GSPLAT_PYTHON -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
" "$INDICES_FILE")

echo "=== gsplat: $SCENE/$METHOD k=$K | Indices: $INDICES ==="

GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_${STRATEGY}

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
        if f.startswith('val') and f.endswith('.json'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            psnr = s.get('psnr', 0)
            ssim = s.get('ssim', 0)
            lpips_val = s.get('lpips', 0)
            print('%s/%s/k%d: PSNR=%.4f SSIM=%.4f LPIPS=%.4f' % ('$SCENE', '$METHOD', $K, psnr, ssim, lpips_val))
" "$GSPLAT_RESULT/stats"

echo "=== Done: $SCENE/$METHOD/k$K ==="

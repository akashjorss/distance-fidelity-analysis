#!/bin/bash
#SBATCH --job-name=v3_nerf
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3n_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v3n_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# NeRF experiments for all methods
# Submit (METHOD = infomax / fvs / lpips_fvs / fvs_euclidean / fvs_angular / fvs_plucker / random):
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=infomax slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=fvs slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=lpips_fvs slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=fvs_euclidean slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=fvs_angular slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=fvs_plucker slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=random_s42 slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=random_s123 slurm/test_v3_nerf.sh
#   sbatch --partition=gpu --cpus-per-task=10 --array=0-15 --export=METHOD=random_s456 slurm/test_v3_nerf.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
NERF_DIR=$WORKDIR/nerf-pytorch
NERF_SCRIPT=$NERF_DIR/run_nerf.py

METHOD=${METHOD:-infomax}

# Map METHOD to results directory and strategy name for index lookup
case $METHOD in
    infomax)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_dinov2_L4
        STRAT_NAME="infomax"
        ;;
    fvs)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3
        STRAT_NAME="fvs"
        ;;
    lpips_fvs)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_lpips
        STRAT_NAME="lpips_fvs"
        ;;
    fvs_euclidean)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_fvs_euclidean
        STRAT_NAME="fvs_euclidean"
        ;;
    fvs_angular)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_fvs_angular
        STRAT_NAME="fvs_angular"
        ;;
    fvs_plucker)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_fvs_plucker
        STRAT_NAME="fvs_plucker"
        ;;
    random_s42)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_random_s42
        STRAT_NAME="random"
        ;;
    random_s123)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_random_s123
        STRAT_NAME="random"
        ;;
    random_s456)
        GSPLAT_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_random_s456
        STRAT_NAME="random"
        ;;
    *)
        echo "Unknown METHOD: $METHOD"
        exit 1
        ;;
esac

NERF_RESULTS=$WORKDIR/ConMax3D_reproduce/results/v3_nerf_${METHOD}

# 16 scenes: 0-7 = LLFF, 8-15 = T&T
SCENES=(fern flower fortress horns leaves orchids room trex \
        Ballroom Barn Church Family Francis Horse Ignatius Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
DATA_FACTORS=(4 4 4 4 4 4 4 4 1 1 1 1 1 1 1 1)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}

echo "=== [$IDX] $SCENE | NeRF $METHOD ==="

# Setup modules
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

source activate $WORKDIR/conda_envs/env_nerf
export PYTHONNOUSERSITE=1

# Read indices from existing JSON
INDICES_FILE="$GSPLAT_RESULTS/$SCENE/train_indices_${SCENE}_${STRAT_NAME}.json"
if [ ! -f "$INDICES_FILE" ]; then
    echo "ERROR: no indices file at $INDICES_FILE"
    exit 1
fi

INDICES=$(python -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
" "$INDICES_FILE")

echo "=== Indices: $INDICES ==="

# Determine flags for T&T vs LLFF
EXTRA_FLAGS=""
if [ $IDX -ge 8 ]; then
    # T&T scenes: non-forward-facing, use no_ndc and spherify
    EXTRA_FLAGS="--no_ndc --spherify"
fi

EXPNAME="${SCENE}_${METHOD}"
RESULT_DIR="$NERF_RESULTS/$SCENE"
mkdir -p "$RESULT_DIR"

cd $NERF_DIR

python $NERF_SCRIPT \
    --datadir "$DATA_BASE/$SCENE" \
    --dataset_type llff \
    --factor $DFACTOR \
    --expname "$EXPNAME" \
    --basedir "$RESULT_DIR" \
    --train_indices "$INDICES" \
    --N_iters 50000 \
    --i_testset 50000 \
    --i_weights 50000 \
    --i_video 100000 \
    --N_samples 64 \
    --N_importance 64 \
    --N_rand 1024 \
    --use_viewdirs \
    --raw_noise_std 1e0 \
    $EXTRA_FLAGS

echo "=== RESULT ==="
METRICS_FILE="$RESULT_DIR/$EXPNAME/testset_050000/metrics.json"
if [ -f "$METRICS_FILE" ]; then
    python -c "
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
print('$SCENE/$METHOD: PSNR=%.4f' % m['psnr'])
" "$METRICS_FILE"
else
    echo "No metrics file found at $METRICS_FILE"
fi

echo "=== Done: $SCENE/$METHOD ==="

#!/bin/bash
#SBATCH --job-name=v2_nerf
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2nerf_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2nerf_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# NOTE: Submit with --array=0-15 --dependency=afterok:<3dgs_jobid>
#   sbatch --array=0-15 slurm/test_v2_full_nerf.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
NERF_TRAINER=$WORKDIR/nerf-pytorch/run_nerf.py
RESULTS_3DGS=$WORKDIR/ConMax3D_reproduce/results/v2_full
RESULTS_NERF=$WORKDIR/ConMax3D_reproduce/results/v2_full_nerf

# 16 scenes: 0-7 = LLFF, 8-15 = T&T
SCENES=(fern flower fortress horns leaves orchids room trex \
        Ballroom Barn Church Family Francis Horse Ignatius Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks \
            $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
# NeRF factors: 8 for LLFF (standard), 4 for T&T (larger images)
NERF_FACTORS=(8 8 8 8 8 8 8 8 4 4 4 4 4 4 4 4)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
NFACTOR=${NERF_FACTORS[$IDX]}

STRATEGIES="submodular_geometric pure_fvs"
N_ITERS=200000

echo "=== [$IDX] $SCENE | NeRF | $N_ITERS iters ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

source activate $WORKDIR/conda_envs/env_nerf
export PYTHONNOUSERSITE=1

for STRAT in $STRATEGIES; do
    INDICES_FILE="$RESULTS_3DGS/$SCENE/train_indices_${SCENE}_${STRAT}.json"
    if [ ! -f "$INDICES_FILE" ]; then
        echo "SKIP $STRAT: no indices file at $INDICES_FILE"
        continue
    fi

    INDICES=$(python -c "
import json
with open('$INDICES_FILE') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")

    EXPNAME="${SCENE}_${STRAT}"
    echo ""
    echo "============================================"
    echo "=== NeRF: $EXPNAME ==="
    echo "=== Indices: $INDICES ==="
    echo "============================================"

    python $NERF_TRAINER \
        --dataset_type llff \
        --datadir $DATA_BASE/$SCENE \
        --expname $EXPNAME \
        --basedir $RESULTS_NERF \
        --factor $NFACTOR \
        --llffhold 8 \
        --N_rand 1024 \
        --N_samples 64 \
        --N_importance 64 \
        --use_viewdirs \
        --raw_noise_std 1e0 \
        --N_iters $N_ITERS \
        --i_testset $N_ITERS \
        --i_weights 50000 \
        --i_print 5000 \
        --train_indices "$INDICES"

    echo "=== $EXPNAME RESULT ==="
    # Extract final test PSNR from metrics JSON
    METRICS_FILE="$RESULTS_NERF/$EXPNAME/metrics_${N_ITERS}.json"
    if [ -f "$METRICS_FILE" ]; then
        python -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)
print(f'$SCENE/$STRAT: PSNR={m[\"psnr\"]:.4f}')
"
    else
        echo "No metrics file found at $METRICS_FILE"
        # Try to find any metrics file
        ls $RESULTS_NERF/$EXPNAME/metrics*.json 2>/dev/null
    fi
done

echo ""
echo "=== ALL DONE: $SCENE (NeRF) ==="

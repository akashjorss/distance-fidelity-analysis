#!/bin/bash
#SBATCH --job-name=nerf_ns
#SBATCH --output=logs/nerf_ns_%A_%a.out
#SBATCH --error=logs/nerf_ns_%A_%a.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-23%8
#SBATCH --export=NONE

set -x

# 8 scenes × 3 methods × k=10 only = 24 runs
SCENES=(chair drums ficus hotdog lego materials mic ship)
METHODS=(conmax3d random fvs)

# Decode array index
IDX=$SLURM_ARRAY_TASK_ID
N_METHODS=${#METHODS[@]}

SCENE_IDX=$((IDX / N_METHODS))
METHOD_IDX=$((IDX % N_METHODS))

SCENE=${SCENES[$SCENE_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
FRAME_COUNT=ten

echo "Job $IDX: scene=$SCENE method=$METHOD budget=$FRAME_COUNT"

WORKDIR=/gpfs/workdir/malhotraa
DATA_DIR=$WORKDIR/data/nerf_synthetic_eschernet/nerf_synthetic/$SCENE
RESULTS_BASE=$WORKDIR/ConMax3D_reproduce/results
RESULT_DIR=$RESULTS_BASE/nerf_ns/$FRAME_COUNT/$METHOD/$SCENE
INDICES_FILE=$RESULTS_BASE/ns/$FRAME_COUNT/$METHOD/train_indices.json
NERF_DIR=$WORKDIR/nerf-pytorch

# Setup environment
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/conmax3d
export PYTHONNOUSERSITE=1

# Read train indices
if [ ! -f "$INDICES_FILE" ]; then
    echo "ERROR: Train indices file not found: $INDICES_FILE"
    exit 1
fi

TRAIN_INDICES=$(python3 -c "
import json
with open('$INDICES_FILE') as f:
    data = json.load(f)
print(','.join(map(str, data['$SCENE'])))
")

if [ -z "$TRAIN_INDICES" ]; then
    echo "ERROR: No train indices found for scene $SCENE"
    exit 1
fi

echo "Train indices: $TRAIN_INDICES"

mkdir -p $RESULT_DIR

START_TIME=$(date +%s)

cd $NERF_DIR

python run_nerf.py \
    --dataset_type blender \
    --datadir $DATA_DIR \
    --basedir $RESULT_DIR \
    --expname ${SCENE}_${METHOD} \
    --white_bkgd \
    --half_res \
    --train_indices "$TRAIN_INDICES" \
    --N_iters 50000 \
    --i_video 999999 --i_testset 50000 \
    --i_weights 50000 \
    --i_print 500 \
    --use_wandb \
    --wandb_project conmax3d-reproduce \
    --wandb_run_name "nerf_ns_${SCENE}_${METHOD}_k10" \
    --no_batching \
    --use_viewdirs \
    --N_importance 128 \
    --N_samples 64

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "NeRF training time: ${ELAPSED}s"

echo "Vanilla NeRF training done for NS: $SCENE $METHOD k=10"

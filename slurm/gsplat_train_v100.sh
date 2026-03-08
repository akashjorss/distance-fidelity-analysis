#!/bin/bash
#SBATCH --job-name=gsplat_train
#SBATCH --output=logs/gsplat_train_%j.out
#SBATCH --error=logs/gsplat_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=NONE

set -x

SCENE=$1
METHOD=$2
FRAME_COUNT=$3

if [ -z "$SCENE" ] || [ -z "$METHOD" ] || [ -z "$FRAME_COUNT" ]; then
    echo "Usage: sbatch gsplat_train.sh <scene> <method> <frame_count>"
    echo "  method: conmax3d, random, fvs"
    echo "  frame_count: ten, twenty, twenty_five"
    exit 1
fi

WORKDIR=/gpfs/workdir/malhotraa
DATA_DIR=$WORKDIR/data/LLFF/$SCENE
RESULTS_BASE=$WORKDIR/ConMax3D_reproduce/results
RESULT_DIR=$RESULTS_BASE/gsplat/$FRAME_COUNT/$METHOD/$SCENE
INDICES_FILE=$RESULTS_BASE/$FRAME_COUNT/$METHOD/train_indices.json

# Setup environment
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1

# Read train indices from JSON
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

echo "Train indices for $SCENE ($METHOD, $FRAME_COUNT): $TRAIN_INDICES"

cd $WORKDIR/gsplat/examples

python simple_trainer.py \
    --data_dir $DATA_DIR \
    --dataset_type colmap \
    --data_factor 4 \
    --train_indices $TRAIN_INDICES \
    --result_dir $RESULT_DIR \
    --max_steps 30000 \
    --eval_steps 7000 15000 30000 \
    --save_steps 30000 \
    --init_type random \
    --disable_viewer

echo "3DGS training done for: $SCENE $METHOD $FRAME_COUNT"

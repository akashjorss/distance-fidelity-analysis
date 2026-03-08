#!/bin/bash
#SBATCH --job-name=gsplat_llff
#SBATCH --output=logs/gsplat_llff_%A_%a.out
#SBATCH --error=logs/gsplat_llff_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-71%4
#SBATCH --export=NONE

set -x

# 8 scenes × 3 methods × 3 budgets = 72 runs
SCENES=(fern flower fortress horns leaves orchids room trex)
METHODS=(conmax3d random fvs)
BUDGETS=(ten twenty twenty_five)

# Decode array index → (scene, method, budget)
IDX=$SLURM_ARRAY_TASK_ID
N_METHODS=${#METHODS[@]}
N_BUDGETS=${#BUDGETS[@]}

SCENE_IDX=$((IDX / (N_METHODS * N_BUDGETS)))
REMAINDER=$((IDX % (N_METHODS * N_BUDGETS)))
METHOD_IDX=$((REMAINDER / N_BUDGETS))
BUDGET_IDX=$((REMAINDER % N_BUDGETS))

SCENE=${SCENES[$SCENE_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
FRAME_COUNT=${BUDGETS[$BUDGET_IDX]}

echo "Job $IDX: scene=$SCENE method=$METHOD budget=$FRAME_COUNT"

WORKDIR=/gpfs/workdir/malhotraa
DATA_DIR=$WORKDIR/data/LLFF/$SCENE
RESULTS_BASE=$WORKDIR/ConMax3D_reproduce/results
RESULT_DIR=$RESULTS_BASE/gsplat/$FRAME_COUNT/$METHOD/$SCENE
INDICES_FILE=$RESULTS_BASE/$FRAME_COUNT/$METHOD/train_indices.json

# Skip if results already exist (fern/room k=10 already done)
if [ -d "$RESULT_DIR/stats" ] && [ "$(ls -A $RESULT_DIR/stats/ 2>/dev/null)" ]; then
    echo "Results already exist at $RESULT_DIR, skipping"
    exit 0
fi

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

if [ -z "$TRAIN_INDICES" ]; then
    echo "ERROR: No train indices found for scene $SCENE in $INDICES_FILE"
    exit 1
fi

echo "Train indices: $TRAIN_INDICES"

START_TIME=$(date +%s)

cd $WORKDIR/gsplat/examples

python simple_trainer.py default \
    --data_dir $DATA_DIR \
    --dataset_type colmap \
    --data_factor 4 \
    --train_indices $TRAIN_INDICES \
    --result_dir $RESULT_DIR \
    --max_steps 30000 \
    --eval_steps 7000 15000 30000 \
    --save_steps 30000 \
    --use_wandb \
    --wandb_project conmax3d-reproduce \
    --wandb_run_name "llff_${SCENE}_${METHOD}_${FRAME_COUNT}" \
    --disable_viewer

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Training time: ${ELAPSED}s"

echo "3DGS training done for: $SCENE $METHOD $FRAME_COUNT"

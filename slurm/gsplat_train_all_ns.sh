#!/bin/bash
#SBATCH --job-name=gsplat_ns
#SBATCH --output=logs/gsplat_ns_%A_%a.out
#SBATCH --error=logs/gsplat_ns_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-71%4
#SBATCH --export=NONE

set -x

# 8 scenes × 3 methods × 3 budgets = 72 runs
SCENES=(chair drums ficus hotdog lego materials mic ship)
METHODS=(conmax3d random fvs)
BUDGETS=(ten twenty twenty_five)

# Decode array index
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
# Preprocessed NeRF Synthetic data (with transforms.json + RGB images/)
DATA_DIR=$WORKDIR/data/nerf_synthetic_gsplat/$SCENE
RESULTS_BASE=$WORKDIR/ConMax3D_reproduce/results
RESULT_DIR=$RESULTS_BASE/gsplat_ns/$FRAME_COUNT/$METHOD/$SCENE
INDICES_FILE=$RESULTS_BASE/ns/$FRAME_COUNT/$METHOD/train_indices.json

# Setup environment
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1

# Check preprocessed data exists
if [ ! -f "$DATA_DIR/transforms.json" ]; then
    echo "ERROR: Preprocessed data not found at $DATA_DIR/transforms.json"
    echo "Run: python scripts/prep_nerf_synthetic.py first"
    exit 1
fi

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
    echo "ERROR: No train indices found for scene $SCENE in $INDICES_FILE"
    exit 1
fi

echo "Train indices: $TRAIN_INDICES"

START_TIME=$(date +%s)

cd $WORKDIR/gsplat/examples

# NeRF Synthetic uses nerfstyle dataset_type with random init (no COLMAP point cloud)
python simple_trainer.py default \
    --data_dir $DATA_DIR \
    --dataset_type nerfstyle \
    --init_type random \
    --data_factor 1 \
    --train_indices $TRAIN_INDICES \
    --result_dir $RESULT_DIR \
    --max_steps 30000 \
    --eval_steps 7000 15000 30000 \
    --save_steps 30000 \
    --use_wandb \
    --wandb_project conmax3d-reproduce \
    --wandb_run_name "ns_${SCENE}_${METHOD}_${FRAME_COUNT}" \
    --disable_viewer

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Training time: ${ELAPSED}s"

echo "3DGS training done for NS: $SCENE $METHOD $FRAME_COUNT"

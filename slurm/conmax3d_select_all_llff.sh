#!/bin/bash
#SBATCH --job-name=conmax3d_llff
#SBATCH --output=logs/conmax3d_select_llff_%A_%a.out
#SBATCH --error=logs/conmax3d_select_llff_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --export=NONE

set -x

# 6 remaining LLFF scenes (fern + room already done)
SCENES=(flower fortress horns leaves orchids trex)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
DATA_DIR=$WORKDIR/data/LLFF
OUTPUT_DIR=$WORKDIR/ConMax3D_reproduce/results

# Setup environment
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/conmax3d
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKDIR/segment-anything-2:$PYTHONPATH

cd $SCRIPTS

python conmax3d_sam2_wandb.py \
    --base_dir $DATA_DIR \
    --scene $SCENE \
    --output_dir $OUTPUT_DIR \
    --num_frames 25 \
    --downscale_factor 4 \
    --pred_iou_thresh 0.8 \
    --use_wandb \
    --wandb_project conmax3d-reproduce

echo "ConMax3D selection done for scene: $SCENE"

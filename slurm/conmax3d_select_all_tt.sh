#!/bin/bash
#SBATCH --job-name=conmax3d_tt
#SBATCH --output=logs/conmax3d_select_tt_%A_%a.out
#SBATCH --error=logs/conmax3d_select_tt_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --export=NONE

set -x

# 8 T&T scenes (note: some have many images, Church ~400)
SCENES=(Ballroom Barn Church Family Francis Horse Ignatius Museum)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
DATA_DIR=$WORKDIR/data/Tanks
OUTPUT_DIR=$WORKDIR/ConMax3D_reproduce/results/tt

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
    --wandb_project conmax3d-reproduce \
    --wandb_run_name "tt_conmax3d_${SCENE}"

echo "ConMax3D selection done for T&T scene: $SCENE"

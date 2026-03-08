#!/bin/bash
#SBATCH --job-name=conmax3d_ns
#SBATCH --output=logs/conmax3d_select_ns_%A_%a.out
#SBATCH --error=logs/conmax3d_select_ns_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --export=NONE

set -x

SCENES=(chair drums ficus hotdog lego materials mic ship)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
DATA_DIR=$WORKDIR/data/nerf_synthetic_eschernet/nerf_synthetic
OUTPUT_DIR=$WORKDIR/ConMax3D_reproduce/results/ns

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
    --downscale_factor 1 \
    --pred_iou_thresh 0.8 \
    --dataset_type nerf_synthetic \
    --use_wandb \
    --wandb_project conmax3d-reproduce \
    --wandb_run_name "ns_conmax3d_${SCENE}"

echo "ConMax3D selection done for NeRF Synthetic scene: $SCENE"

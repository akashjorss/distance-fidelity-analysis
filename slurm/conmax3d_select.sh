#!/bin/bash
#SBATCH --job-name=conmax3d_select
#SBATCH --output=logs/conmax3d_select_%j.out
#SBATCH --error=logs/conmax3d_select_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

set -x

SCENE=$1
if [ -z "$SCENE" ]; then
    echo "Usage: sbatch conmax3d_select.sh <scene_name>"
    exit 1
fi

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

# Add SAM2 to path
export PYTHONPATH=$WORKDIR/segment-anything-2:$PYTHONPATH

cd $SCRIPTS

python conmax3d_sam2_wandb.py \
    --base_dir $DATA_DIR \
    --scene $SCENE \
    --output_dir $OUTPUT_DIR \
    --num_frames 25 \
    --use_wandb \
    --wandb_project conmax3d-reproduction

echo "ConMax3D selection done for scene: $SCENE"

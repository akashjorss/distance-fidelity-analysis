#!/bin/bash
#SBATCH --job-name=masks_segmentation
#SBATCH --output=slurm_outputs/masks_segmentation_%j.out
#SBATCH --error=slurm_outputs/masks_segmentation_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 DATASET SCENE MODEL"
    exit 1
fi

# Assign command-line arguments to variables
DATASET=$1
SCENE=$2
MODEL=$3

# Print the variables to verify
echo "Dataset: $DATASET"
echo "Scene: $SCENE"
echo "Model: $MODEL"

# Load the environment
source $WORKDIR/load_modules_sam2.sh
python -c "import torch; print('cuda is available: ', torch.cuda.is_available())"

echo 'environment sam 2 loaded successfully'

cd /gpfs/workdir/malhotraa/masks_segmentation

echo 'running segmentation'
python conmax3d_sam2.py /gpfs/workdir/malhotraa/data/$DATASET $SCENE /gpfs/workdir/malhotraa/ConMax3D_experiments/$MODEL/$DATASET

# sbatch masks_segmentation_slurm.sh <DATASET> <SCENE> <MODEL>
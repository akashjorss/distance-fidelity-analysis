#!/bin/bash
#SBATCH --job-name=feat
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/feat_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/feat_%A_%a.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# Feature extraction (DINOv2, AlexNet, CLIP) for all datasets
# Submit:
#   sbatch --array=0-2 slurm/submit_features.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts

DATASETS=(llff tt ns)
IDX=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$IDX]}

echo "=== Feature extraction: $DATASET ==="

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

python $SCRIPTS/extract_features.py --dataset $DATASET --model all --batch_size 8

echo "=== Done: $DATASET ==="

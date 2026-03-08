#!/bin/bash
#SBATCH --job-name=entropy
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/ent_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/ent_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
# Entropy analysis: swap local search + entropy-PSNR correlation
# Submit:
#   sbatch --partition=gpu_test --cpus-per-task=10 --array=0-7 slurm/test_entropy_analysis.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
RESULTS=$WORKDIR/ConMax3D_reproduce/results
OUTPUT=$RESULTS/entropy_analysis

SCENES=(fern flower fortress horns leaves orchids room trex)
IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}

echo "=== [$IDX] $SCENE | Entropy Analysis ==="

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

python $SCRIPTS/entropy_analysis.py \
    --data_dir $WORKDIR/data/LLFF \
    --scene $SCENE \
    --results_base $RESULTS \
    --output_dir $OUTPUT \
    --data_factor 4 \
    --layer 4 \
    --batch_size 8

echo "=== Done: $SCENE ==="

#!/bin/bash
#SBATCH --job-name=conmax3d_sweep
#SBATCH --output=logs/conmax3d_sweep_%A_%a.out
#SBATCH --error=logs/conmax3d_sweep_%A_%a.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --export=NONE

set -x

# Usage: sbatch conmax3d_sweep.sh <sweep_id> <dataset> [count_per_agent]
SWEEP_ID=$1
SWEEP_DATASET=${2:-llff}
COUNT=${3:-15}

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch conmax3d_sweep.sh <sweep_id> <dataset> [count_per_agent]"
    exit 1
fi

WORKDIR=/gpfs/workdir/malhotraa

# Setup modules
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# Activate conmax3d env (for ConMax3D selection + wandb agent)
source activate $WORKDIR/conda_envs/conmax3d

# Only SAM2 in PYTHONPATH (NOT gsplat — it conflicts with gsplat_env)
export PYTHONPATH=$WORKDIR/segment-anything-2:$PYTHONPATH
export PYTHONNOUSERSITE=1
export SWEEP_DATASET=$SWEEP_DATASET
export TORCH_CUDA_ARCH_LIST="7.0 8.0"

# gsplat_env/bin AFTER conmax3d/bin so 'python' still resolves to conmax3d
export PATH=$PATH:$WORKDIR/conda_envs/gsplat_env/bin

cd $WORKDIR/ConMax3D_reproduce

echo "Sweep agent $SLURM_ARRAY_TASK_ID: dataset=$SWEEP_DATASET, $COUNT trials"
wandb agent --count $COUNT akashjorss/conmax3d-sweep/$SWEEP_ID

echo "Sweep agent $SLURM_ARRAY_TASK_ID done"

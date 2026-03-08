#!/bin/bash
#SBATCH --job-name=conmax3d_sweep
#SBATCH --output=logs/conmax3d_sweep_%j.out
#SBATCH --error=logs/conmax3d_sweep_%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=NONE

set -x

SWEEP_ID=$1
COUNT=${2:-5}

if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch wandb_sweep_agent.sh <sweep_id> [count]"
    exit 1
fi

WORKDIR=/gpfs/workdir/malhotraa

# Setup
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# Need both envs accessible - use conmax3d as primary
source activate $WORKDIR/conda_envs/conmax3d
export PYTHONPATH=$WORKDIR/segment-anything-2:$WORKDIR/gsplat:$PYTHONPATH

cd $WORKDIR/ConMax3D_reproduce

wandb agent --count $COUNT $SWEEP_ID

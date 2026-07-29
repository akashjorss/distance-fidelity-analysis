#!/bin/bash
#SBATCH --job-name=nerf_metrics
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/metrics_%j.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/metrics_%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=NONE

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate /gpfs/workdir/malhotraa/conda_envs/env_nerf

cd /gpfs/workdir/malhotraa/ConMax3D_reproduce
set -x
python compute_nerf_metrics.py

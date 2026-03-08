#!/bin/bash
#SBATCH --job-name=debug_gsplat
#SBATCH --output=logs/debug_gsplat_%j.out
#SBATCH --error=logs/debug_gsplat_%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --export=NONE

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed "s/ /:/g")
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate /gpfs/workdir/malhotraa/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1

cd /gpfs/workdir/malhotraa/gsplat/examples

# Quick test: just 10 steps
python simple_trainer.py default \
    --data_dir /gpfs/workdir/malhotraa/data/LLFF/fern \
    --dataset_type colmap \
    --data_factor 4 \
    --train_indices 0,1,2,3,4,5,6,7,8,9 \
    --result_dir /tmp/debug_gsplat \
    --max_steps 10 \
    --eval_steps 10 \
    --save_steps 10 \
    --init_type random \
    --disable_viewer

echo "EXIT CODE: $?"

#!/bin/bash
#SBATCH --job-name=debug_gs3
#SBATCH --output=logs/debug_gs3_%j.out
#SBATCH --error=logs/debug_gs3_%j.err
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --export=NONE

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate /gpfs/workdir/malhotraa/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1

cd /gpfs/workdir/malhotraa/gsplat/examples

echo "=== Test 1: NO train_indices ==="
python simple_trainer.py default \
    --data_dir /gpfs/workdir/malhotraa/data/LLFF/fern \
    --dataset_type colmap \
    --data_factor 4 \
    --result_dir /tmp/debug_gsplat3 \
    --max_steps 50 \
    --eval_steps 50 \
    --save_steps 50 \
    --init_type random \
    --disable_viewer
echo "Test 1 exit: $?"

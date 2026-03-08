#!/bin/bash
#SBATCH --job-name=dust3r_ns
#SBATCH --output=logs/dust3r_ns_%A_%a.out
#SBATCH --error=logs/dust3r_ns_%A_%a.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --export=NONE

set -x

SCENES=(chair drums ficus hotdog lego materials mic ship)
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

WORKDIR=/gpfs/workdir/malhotraa

# Setup
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

cd $WORKDIR/ConMax3D_reproduce

echo "Processing scene: $SCENE"
python scripts/preprocess_dust3r_ns.py \
  --data_dir $WORKDIR/data/nerf_synthetic_gsplat/$SCENE \
  --dust3r_path $WORKDIR/stable-virtual-camera/third_party/dust3r \
  --seva_path $WORKDIR/stable-virtual-camera \
  --num_images 30 \
  --conf_threshold 1.5 \
  --batch_size 8

echo "Done: $SCENE"
seff $SLURM_JOB_ID

#!/bin/bash
#SBATCH --job-name=pfid
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/pfid_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/pfid_%A_%a.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# Per-frame fidelity extraction from gsplat checkpoints
# Submit:
#   sbatch --array=0-24 slurm/submit_fidelity.sh
#   SCENE=fern sbatch --array=0 slurm/submit_fidelity.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts

# All experiments with checkpoints
EXPERIMENTS=(
    v3_k15_random
    v3_k15_infomax
    v3_k15_fvs
    v3_k15_fvs_plucker
    v3_k20_random
    v3_k20_infomax
    v3_k20_fvs
    v3_k20_fvs_plucker
    v3_k25_random
    v3_k25_infomax
    v3_k25_fvs
    v3_k25_fvs_plucker
    v3_random_s42
    v3_random_s123
    v3_random_s456
    v3_fvs_angular
    v3_fvs_euclidean
    v3_fvs_plucker
    v3_swap
    v3_dinov2_L2+4+6+8_concat
    v3_dinov2_L4+8_concat
    v3_dinov2_L4+6+8+10_concat
    v3_dinov2_L2+4+6_concat
    v3_dinov2_L0+2+4+6+8+10_concat
    v3_dinov2_L0+1+2+3+4+5+6+7+8+9+10+11_concat
)

IDX=$SLURM_ARRAY_TASK_ID
EXPERIMENT=${EXPERIMENTS[$IDX]}

echo "=== Per-frame fidelity: $EXPERIMENT ==="

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

if [ -n "$SCENE" ]; then
    python $SCRIPTS/extract_perframe_fidelity.py --experiment $EXPERIMENT --scene $SCENE
else
    python $SCRIPTS/extract_perframe_fidelity.py --experiment $EXPERIMENT
fi

echo "=== Done: $EXPERIMENT ==="

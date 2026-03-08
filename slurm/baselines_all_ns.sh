#!/bin/bash
#SBATCH --job-name=baselines_ns
#SBATCH --output=logs/baselines_ns_%j.out
#SBATCH --error=logs/baselines_ns_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NONE

set -x

WORKDIR=/gpfs/workdir/malhotraa
DATA_DIR=$WORKDIR/data/nerf_synthetic_eschernet/nerf_synthetic
OUTPUT_DIR=$WORKDIR/ConMax3D_reproduce/results/ns
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/conmax3d
export PYTHONNOUSERSITE=1

SCENES=(chair drums ficus hotdog lego materials mic ship)

for SCENE in "${SCENES[@]}"; do
    for METHOD in random fvs; do
        echo "=== NS $SCENE / $METHOD ==="
        python $SCRIPTS/baselines.py \
            --data_dir $DATA_DIR/$SCENE \
            --scene $SCENE \
            --output_dir $OUTPUT_DIR \
            --method $METHOD \
            --dataset_type nerf_synthetic
    done
done

echo "All NeRF Synthetic baselines done"

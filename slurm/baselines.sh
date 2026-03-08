#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NONE

set -x

WORKDIR=/gpfs/workdir/malhotraa
DATA_DIR=$WORKDIR/data/LLFF
OUTPUT_DIR=$WORKDIR/ConMax3D_reproduce/results
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/conmax3d

for SCENE in fern room; do
    for METHOD in random fvs; do
        python $SCRIPTS/baselines.py \
            --data_dir $DATA_DIR/$SCENE \
            --scene $SCENE \
            --output_dir $OUTPUT_DIR \
            --method $METHOD
    done
done

echo "All baselines done"

#!/bin/bash
#SBATCH --job-name=viz_feat
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/viz_%j.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/viz_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --export=NONE

set -ex
WORKDIR=/gpfs/workdir/malhotraa

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
OUTPUT=$WORKDIR/ConMax3D_reproduce/results/feature_viz

python $SCRIPTS/visualize_features.py $WORKDIR/data/LLFF/horns $OUTPUT/horns
python $SCRIPTS/visualize_features.py $WORKDIR/data/LLFF/room $OUTPUT/room
python $SCRIPTS/visualize_features.py $WORKDIR/data/Tanks/Church $OUTPUT/Church

echo "=== Done ==="
ls -la $OUTPUT/horns/ $OUTPUT/room/ $OUTPUT/Church/

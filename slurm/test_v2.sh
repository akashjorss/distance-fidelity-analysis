#!/bin/bash
#SBATCH --job-name=conmax3d_v2
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ruche-gpu18
#SBATCH --export=NONE

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v2
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

# Scene configs: scene_name  data_base_dir  data_factor  max_images
SCENES=(fern trex Church Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
DATA_FACTORS=(4 4 1 1)
MAX_IMAGES=(0 0 150 0)  # 0 = no limit

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}
MAXIMG=${MAX_IMAGES[$IDX]}

K=10
echo "=== Scene: $SCENE (data_factor=$DFACTOR, max_images=$MAXIMG) ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# ── Phase 1: Frame selection (seva env with SAM2) ──
echo "=== Phase 1: ConMax3D v2 frame selection ==="
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WORKDIR/segment-anything-2:$PYTHONPATH

MAX_IMAGES_FLAG=""
if [ "$MAXIMG" -gt 0 ]; then
    MAX_IMAGES_FLAG="--max_images $MAXIMG"
fi

python $SCRIPTS/conmax3d_v2.py \
    --data_dir $DATA_BASE \
    --scene $SCENE \
    --output_dir $RESULTS/$SCENE \
    --k $K \
    --data_factor $DFACTOR \
    --n_concepts $((K * 2)) \
    $MAX_IMAGES_FLAG

# Read selected indices
INDICES=$(python -c "
import json
with open('$RESULTS/$SCENE/train_indices_$SCENE.json') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")
echo "Selected indices: $INDICES"

# ── Phase 2: gsplat 3DGS training (gsplat_env) ──
echo "=== Phase 2: gsplat training ==="
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1

# Add gsplat_env bin to PATH for ninja etc.
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

GSPLAT_RESULT=$RESULTS/$SCENE/gsplat

$GSPLAT_PYTHON $GSPLAT_TRAINER default \
    --data_dir $DATA_BASE/$SCENE \
    --dataset_type colmap \
    --data_factor $DFACTOR \
    --init_type sfm \
    --train_indices "$INDICES" \
    --result_dir $GSPLAT_RESULT \
    --max_steps 30000 \
    --eval_steps 30000 \
    --save_steps 30000 \
    --disable_viewer

# ── Extract PSNR ──
echo "=== Results ==="
python -c "
import json, glob, os
stats_dir = '$GSPLAT_RESULT/stats'
if os.path.isdir(stats_dir):
    for f in sorted(os.listdir(stats_dir)):
        if f.endswith('.json'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            print(f'$SCENE: PSNR={s.get(\"psnr\", \"N/A\"):.4f} SSIM={s.get(\"ssim\", \"N/A\"):.4f} LPIPS={s.get(\"lpips\", \"N/A\"):.4f}')
else:
    print('No stats directory found')
"

echo "=== Done: $SCENE ==="

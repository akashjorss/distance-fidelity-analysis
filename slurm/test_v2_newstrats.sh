#!/bin/bash
#SBATCH --job-name=v2_newstrats
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2ns_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2ns_%A_%a.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v2_newstrats
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

SCENES=(fern trex Church Museum)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/Tanks $WORKDIR/data/Tanks)
DATA_FACTORS=(4 4 1 1)
MAX_IMAGES=(0 0 150 0)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}
MAXIMG=${MAX_IMAGES[$IDX]}

K=10
THRESH=20
# All new strategies + pure_fvs as reference
STRATEGIES="soft_concept_bonus,joint_space_fvs,adaptive_concept_fvs,submodular_geometric,pure_fvs"

echo "=== Scene: $SCENE | New strategies sweep ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

# ── Phase 1: SAM2 + concepts + all strategies (seva env) ──
echo "=== Phase 1: ConMax3D v2 new strategies ==="
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
    --n_concepts 0 \
    --epipolar_threshold $THRESH \
    --strategies $STRATEGIES \
    $MAX_IMAGES_FLAG

# ── Phase 2: gsplat for each strategy (gsplat_env) ──
source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

IFS=',' read -ra STRAT_ARRAY <<< "$STRATEGIES"
for STRAT in "${STRAT_ARRAY[@]}"; do
    INDICES_FILE="$RESULTS/$SCENE/train_indices_${SCENE}_${STRAT}.json"
    if [ ! -f "$INDICES_FILE" ]; then
        echo "SKIP $STRAT: no indices file"
        continue
    fi

    INDICES=$(python -c "
import json
with open('$INDICES_FILE') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")
    echo ""
    echo "============================================"
    echo "=== Phase 2: gsplat for $STRAT ==="
    echo "=== Indices: $INDICES ==="
    echo "============================================"

    GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_$STRAT

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

    echo "=== $STRAT RESULT ==="
    python -c "
import json, os
stats_dir = '$GSPLAT_RESULT/stats'
if os.path.isdir(stats_dir):
    for f in sorted(os.listdir(stats_dir)):
        if f.endswith('.json'):
            with open(os.path.join(stats_dir, f)) as fh:
                s = json.load(fh)
            psnr = s.get('psnr', 0)
            ssim = s.get('ssim', 0)
            lpips = s.get('lpips', 0)
            print(f'$SCENE/$STRAT: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}')
else:
    print('No stats found for $STRAT')
"
done

echo ""
echo "=== ALL DONE: $SCENE ==="

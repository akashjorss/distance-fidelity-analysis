#!/bin/bash
#SBATCH --job-name=v2_gsful
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2gsf_%A_%a.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/v2gsf_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# gsplat-only Phase 2 for failed full eval LLFF jobs
# Submit:
#   sbatch --partition=gpua100 --cpus-per-task=8 --array=0-3 slurm/gsplat_only_full.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
RESULTS=$WORKDIR/ConMax3D_reproduce/results/v2_full
GSPLAT_TRAINER=$WORKDIR/gsplat/examples/simple_trainer.py
GSPLAT_PYTHON=$WORKDIR/conda_envs/gsplat_env/bin/python

# Failed scenes: flower(1), fortress(2), horns(3-pure_fvs only), leaves(4)
SCENES=(flower fortress horns leaves)
DATA_BASES=($WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF $WORKDIR/data/LLFF)
DATA_FACTORS=(4 4 4 4)

IDX=$SLURM_ARRAY_TASK_ID
SCENE=${SCENES[$IDX]}
DATA_BASE=${DATA_BASES[$IDX]}
DFACTOR=${DATA_FACTORS[$IDX]}

echo "=== gsplat-only full eval: $SCENE ==="

# ── Setup modules ──
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0

source activate $WORKDIR/conda_envs/gsplat_env
export PYTHONNOUSERSITE=1
export PATH=$WORKDIR/conda_envs/gsplat_env/bin:$PATH

STRATEGIES="submodular_geometric pure_fvs"
for STRAT in $STRATEGIES; do
    INDICES_FILE="$RESULTS/$SCENE/train_indices_${SCENE}_${STRAT}.json"
    if [ ! -f "$INDICES_FILE" ]; then
        echo "SKIP $STRAT: no indices file"
        continue
    fi

    GSPLAT_RESULT=$RESULTS/$SCENE/gsplat_$STRAT
    # Skip if already completed
    if [ -d "$GSPLAT_RESULT/stats" ] && ls "$GSPLAT_RESULT/stats/"*.json >/dev/null 2>&1; then
        echo "SKIP $STRAT: already has results"
        python -c "
import json, os
stats_dir = '$GSPLAT_RESULT/stats'
for f in sorted(os.listdir(stats_dir)):
    if f.endswith('.json'):
        with open(os.path.join(stats_dir, f)) as fh:
            s = json.load(fh)
        psnr = s.get('psnr', 0)
        ssim = s.get('ssim', 0)
        lpips = s.get('lpips', 0)
        print(f'$SCENE/$STRAT: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}')
"
        continue
    fi

    INDICES=$(python -c "
import json
with open('$INDICES_FILE') as f:
    d = json.load(f)
print(','.join(map(str, d['selected_indices'])))
")
    echo "=== gsplat: $SCENE/$STRAT | Indices: $INDICES ==="

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

    echo "=== $SCENE/$STRAT RESULT ==="
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
"
done

echo "=== Done: $SCENE ==="

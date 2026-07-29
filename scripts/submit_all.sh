#!/bin/bash
# Master SLURM submission script with dependency chains
# Usage: bash scripts/submit_all.sh
#
# Execution order:
#   Phase 1 (LLFF):    1A + 1B → 1C
#   Phase 2 (T&T):     2A + 2B → 2C
#   Phase 3 (NS):      3_prep → 3A + 3B → 3D + 3E
#   Phase 4 (Sweep):   independent
#   All → Phase 5 (manual: python scripts/collect_results.py)

set -e

WORKDIR=/gpfs/workdir/malhotraa
PROJ=$WORKDIR/ConMax3D_reproduce
cd $PROJ

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "ConMax3D Scale-Up: Submitting all phases"
echo "=========================================="

# =============================================
# Phase 0: Preprocess NeRF Synthetic for gsplat
# =============================================
echo ""
echo "Phase 0: Preprocessing NeRF Synthetic..."
source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/conmax3d

python scripts/prep_nerf_synthetic.py \
    --data_dir $WORKDIR/data/nerf_synthetic_eschernet/nerf_synthetic \
    --output_dir $WORKDIR/data/nerf_synthetic_gsplat

echo "Preprocessing done."

# =============================================
# Phase 1: LLFF
# =============================================
echo ""
echo "Phase 1: LLFF"

# 1A: ConMax3D selection (6 remaining scenes)
JOB_1A=$(sbatch --parsable slurm/conmax3d_select_all_llff.sh)
echo "  1A ConMax3D selection: $JOB_1A"

# 1B: Baselines (all 8 scenes)
JOB_1B=$(sbatch --parsable slurm/baselines_all_llff.sh)
echo "  1B Baselines: $JOB_1B"

# 1C: gsplat training (depends on 1A + 1B)
JOB_1C=$(sbatch --parsable --dependency=afterok:${JOB_1A}:${JOB_1B} slurm/gsplat_train_all_llff.sh)
echo "  1C gsplat training: $JOB_1C (depends on $JOB_1A, $JOB_1B)"

# =============================================
# Phase 2: Tanks & Temples
# =============================================
echo ""
echo "Phase 2: Tanks & Temples"

# 2A: ConMax3D selection (8 scenes)
JOB_2A=$(sbatch --parsable slurm/conmax3d_select_all_tt.sh)
echo "  2A ConMax3D selection: $JOB_2A"

# 2B: Baselines (8 scenes)
JOB_2B=$(sbatch --parsable slurm/baselines_all_tt.sh)
echo "  2B Baselines: $JOB_2B"

# 2C: gsplat training (depends on 2A + 2B)
JOB_2C=$(sbatch --parsable --dependency=afterok:${JOB_2A}:${JOB_2B} slurm/gsplat_train_all_tt.sh)
echo "  2C gsplat training: $JOB_2C (depends on $JOB_2A, $JOB_2B)"

# =============================================
# Phase 3: NeRF Synthetic
# =============================================
echo ""
echo "Phase 3: NeRF Synthetic"

# 3A: ConMax3D selection (8 scenes)
JOB_3A=$(sbatch --parsable slurm/conmax3d_select_all_ns.sh)
echo "  3A ConMax3D selection: $JOB_3A"

# 3B: Baselines (8 scenes)
JOB_3B=$(sbatch --parsable slurm/baselines_all_ns.sh)
echo "  3B Baselines: $JOB_3B"

# 3D: gsplat training (depends on 3A + 3B)
JOB_3D=$(sbatch --parsable --dependency=afterok:${JOB_3A}:${JOB_3B} slurm/gsplat_train_all_ns.sh)
echo "  3D gsplat 3DGS training: $JOB_3D (depends on $JOB_3A, $JOB_3B)"

# 3E: Vanilla NeRF training (depends on 3A + 3B)
JOB_3E=$(sbatch --parsable --dependency=afterok:${JOB_3A}:${JOB_3B} slurm/nerf_train_ns.sh)
echo "  3E Vanilla NeRF training: $JOB_3E (depends on $JOB_3A, $JOB_3B)"

# =============================================
# Phase 4: W&B Sweep (independent)
# =============================================
echo ""
echo "Phase 4: W&B Sweep"
echo "  NOTE: Sweep must be created manually first:"
echo "    cd $PROJ && wandb sweep sweep_config_bayesian.yaml"
echo "    Then: sbatch slurm/conmax3d_sweep.sh <SWEEP_ID>"
echo "  Skipping automatic submission (needs sweep_id)"

# =============================================
# Summary
# =============================================
echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Phase 1 (LLFF):     $JOB_1A, $JOB_1B -> $JOB_1C"
echo "Phase 2 (T&T):      $JOB_2A, $JOB_2B -> $JOB_2C"
echo "Phase 3 (NS):       $JOB_3A, $JOB_3B -> $JOB_3D, $JOB_3E"
echo "Phase 4 (Sweep):    manual"
echo ""
echo "Monitor: squeue -u \$USER"
echo "After completion: python scripts/collect_results.py"

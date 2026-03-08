#!/bin/bash
# CCIS Analysis Pipeline Orchestrator
# Submits all jobs with proper dependency chains.
#
# Usage:
#   bash slurm/run_all.sh              # Submit everything
#   bash slurm/run_all.sh --dry-run    # Show commands without submitting

set -e

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

# Extract numeric job ID from sbatch output "Submitted batch job XXXXX"
get_job_id() {
    local output="$1"
    echo "$output" | grep -o '[0-9]\+' | tail -1
}

echo "=== Step 1: Per-frame fidelity extraction (25 experiments) ==="
if $DRY_RUN; then
    echo "  [DRY] sbatch --array=0-24 $SLURM_DIR/submit_fidelity.sh"
    FIDELITY_JOB=100001
else
    FIDELITY_OUT=$(sbatch --array=0-24 $SLURM_DIR/submit_fidelity.sh 2>&1)
    FIDELITY_JOB=$(get_job_id "$FIDELITY_OUT")
    echo "  $FIDELITY_OUT"
fi
echo "  Job ID: $FIDELITY_JOB"

echo ""
echo "=== Step 2: Feature extraction (3 datasets, parallel with Step 1) ==="
if $DRY_RUN; then
    echo "  [DRY] sbatch --array=0-2 $SLURM_DIR/submit_features.sh"
    FEATURES_JOB=100002
else
    FEATURES_OUT=$(sbatch --array=0-2 $SLURM_DIR/submit_features.sh 2>&1)
    FEATURES_JOB=$(get_job_id "$FEATURES_OUT")
    echo "  $FEATURES_OUT"
fi
echo "  Job ID: $FEATURES_JOB"

echo ""
echo "=== Step 3: Distance computation (depends on Step 2: $FEATURES_JOB) ==="
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:$FEATURES_JOB --array=0-24 $SLURM_DIR/submit_distances.sh"
    DISTANCES_JOB=100003
else
    DISTANCES_OUT=$(sbatch --dependency=afterok:$FEATURES_JOB --array=0-24 $SLURM_DIR/submit_distances.sh 2>&1)
    DISTANCES_JOB=$(get_job_id "$DISTANCES_OUT")
    echo "  $DISTANCES_OUT"
fi
echo "  Job ID: $DISTANCES_JOB"

echo ""
echo "=== Step 4+5: Merge + Analysis (depends on Steps 1,3: $FIDELITY_JOB,$DISTANCES_JOB) ==="
if $DRY_RUN; then
    echo "  [DRY] sbatch --dependency=afterok:${FIDELITY_JOB}:${DISTANCES_JOB} $SLURM_DIR/submit_analysis.sh"
    ANALYSIS_JOB=100004
else
    ANALYSIS_OUT=$(sbatch --dependency=afterok:${FIDELITY_JOB}:${DISTANCES_JOB} $SLURM_DIR/submit_analysis.sh 2>&1)
    ANALYSIS_JOB=$(get_job_id "$ANALYSIS_OUT")
    echo "  $ANALYSIS_OUT"
fi
echo "  Analysis job: $ANALYSIS_JOB"

echo ""
echo "=========================================="
echo "  Pipeline submitted successfully!"
echo "=========================================="
echo "  Step 1 (fidelity):   $FIDELITY_JOB  (25 array tasks)"
echo "  Step 2 (features):   $FEATURES_JOB  (3 array tasks)"
echo "  Step 3 (distances):  $DISTANCES_JOB  (25 array tasks, after $FEATURES_JOB)"
echo "  Step 4+5 (analysis): $ANALYSIS_JOB  (after $FIDELITY_JOB + $DISTANCES_JOB)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ls /gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/"

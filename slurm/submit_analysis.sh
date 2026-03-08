#!/bin/bash
#SBATCH --job-name=ccis_analysis
#SBATCH --output=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/analysis_%j.out
#SBATCH --error=/gpfs/workdir/malhotraa/ConMax3D_reproduce/logs/analysis_%j.err
#SBATCH --partition=gpu,gpua100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=ruche-gpu18,ruche-gpu13
#SBATCH --export=NONE
# Run full analysis pipeline after data is merged
# Submit:
#   sbatch slurm/submit_analysis.sh

set -ex

WORKDIR=/gpfs/workdir/malhotraa
SCRIPTS=$WORKDIR/ConMax3D_reproduce/scripts
ANALYSIS=$WORKDIR/ConMax3D_reproduce/analysis
CSV=$WORKDIR/ConMax3D_reproduce/results/perframe/combined_perframe.csv

source /etc/profile.d/lmod.sh
export MODULEPATH=$(ls -d /gpfs/softs/modules/modulefiles/* | xargs | sed 's/ /:/g')
module load cuda/11.8.0/gcc-11.2.0
module load anaconda3/2022.10/gcc-11.2.0
source activate $WORKDIR/conda_envs/seva
export PYTHONNOUSERSITE=1

# Step 4: Merge per-frame data
echo "=== Step 4: Merging per-frame data ==="
python $SCRIPTS/merge_perframe_data.py

# Step 5: Analysis pipeline
echo "=== Step 5a: Correlation analysis ==="
python $ANALYSIS/correlation.py --csv $CSV

echo "=== Step 5b: XGBoost regression ==="
python $ANALYSIS/xgboost_regression.py --csv $CSV --target psnr
python $ANALYSIS/xgboost_regression.py --csv $CSV --target ssim
python $ANALYSIS/xgboost_regression.py --csv $CSV --target lpips

echo "=== Step 5c: SHAP analysis ==="
python $ANALYSIS/shap_analysis.py --csv $CSV --target psnr

echo "=== Step 5d: Binary classification ==="
python $ANALYSIS/binary_classification.py --csv $CSV

echo "=== Step 5e: LOSO generalization ==="
python $ANALYSIS/loso_generalization.py --csv $CSV --target psnr

echo "=== Step 5f: Cross-dataset generalization ==="
python $ANALYSIS/cross_dataset_generalization.py --csv $CSV --target psnr

echo "=== Step 5g: Comprehensive report with qualitative examples ==="
python $ANALYSIS/generate_report.py --csv $CSV --target psnr

echo "=== ALL ANALYSIS DONE ==="

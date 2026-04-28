#!/bin/bash
#SBATCH -J checkpoint_eval
#SBATCH --account=cs5814
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/checkpoint_eval_%A_%a.out
#SBATCH --error=logs/checkpoint_eval_%A_%a.err
#SBATCH --mail-user=avygaur@vt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

mkdir -p logs results

module purge
source venv/bin/activate

# All 12 checkpoints visible in your models/ directory.
# The Python evaluator auto-detects:
#   DEEP in filename          -> DeepFiLM from scripts/deepFilm.py
#   FiLM but not DEEP         -> ShallowFiLM from scripts/models.py
#   no FiLM in filename       -> plain baseline SwinUNETR
CHECKPOINTS=(
    "models/model_checkpoint_FiLM_DEEP0_0.87.pt"
    "models/model_checkpoint_FiLM_DEEP1_0.87.pt"
    "models/model_checkpoint_FiLM_DEEP2_0.87.pt"
    "models/model_checkpoint_FiLM_DEEP3_0.87.pt"
    "models/model_checkpoint_FiLM0_0.55.pt"
    "models/model_checkpoint_FiLM9_0.55.pt"
    "models/model_checkpoint0_0.03.pt"
    "models/model_checkpoint0_0.29.pt"
    "models/model_checkpoint0_0.47.pt"
    "models/model_checkpoint4_0.85.pt"
    "models/model_checkpoint9_0.87.pt"
)

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh models/*.pt || true
    exit 1
fi

echo "=================================================="
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM array task ID: $SLURM_ARRAY_TASK_ID"
echo "Checkpoint:          $CHECKPOINT"
echo "Hostname:            $(hostname)"
echo "Start time:          $(date)"
echo "=================================================="

python scripts/evaluate_checkpoint.py \
    --checkpoint     "$CHECKPOINT" \
    --data_dir       ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --json_list      ./brats23_folds.json \
    --baseline_cache ./dropout_results_cache.json \
    --fold           1 \
    --roi            128 128 128 \
    --sw_batch_size  4 \
    --infer_overlap  0.5 \
    --device         cuda \
    --output_dir     results

echo "=================================================="
echo "Finished checkpoint: $CHECKPOINT"
echo "End time:            $(date)"
echo "=================================================="
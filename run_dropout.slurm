#!/bin/bash
#SBATCH -J brats_dropout
#SBATCH --account=cs5814
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/dropout_%j.out
#SBATCH --error=logs/dropout_%j.err
#SBATCH --mail-user=avygaur@vt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

mkdir -p logs

module purge
source venv/bin/activate

python evaluate_dropout.py \
  --checkpoint  model_checkpoint29_0.55.pt \
  --data_dir    ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --json_list   ./brats23_folds.json \
  --fold        1 \
  --roi         128 128 128 \
  --sw_batch_size 4 \
  --infer_overlap 0.5 \
  --device      cuda \
  --output_plot dropout_results.png
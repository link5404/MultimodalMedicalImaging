python scripts/evaluate_dropout.py \
  --checkpoint  scripts/model.pt \
  --data_dir    dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --json_list   ./brats23_folds.json \
  --fold        1
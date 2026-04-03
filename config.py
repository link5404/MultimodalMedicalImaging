# config.py

# ── Paths (edit these) ──────────────────────────────────────────────────────
VAL_FRACTION= 0.2
DATA_DIR       = "dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" 
CHECKPOINT_DIR  = "/model/checkpoint"
PRETRAINED_WEIGHTS = "/path/to/swin_unetr_pretrained.pt"  # optional

# ── Modalities ───────────────────────────────────────────────────────────────
MODALITIES = ["t1", "t1ce", "t2", "flair"]        # adjust to your dataset naming
NUM_CLASSES = 3                                    # BraTS: TC, WT, ET

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 1
MAX_EPOCHS      = 300
LEARNING_RATE   = 1e-4
VAL_INTERVAL    = 5
ROI_SIZE        = (128, 128, 128)

# ── Missing-modality masking ─────────────────────────────────────────────────
MASK_PROB       = 0.5   # probability of masking at least one modality per sample
MAX_MASKED      = 2     # max modalities to mask in one sample

# ── Loss ─────────────────────────────────────────────────────────────────────
LOSS_TYPE = "dice_ce"   # "dice" | "dice_ce"
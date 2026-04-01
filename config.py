# config.py

# ── Paths (edit these) ──────────────────────────────────────────────────────
DATA_DIR        = "/path/to/your/dataset"          # root of CFB-GBM or BraTS
TRAIN_JSON      = "/path/to/train_split.json"      # {"training": [{"image": [...], "label": "..."}]}
VAL_JSON        = "/path/to/val_split.json"
CHECKPOINT_DIR  = "/path/to/checkpoints"
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
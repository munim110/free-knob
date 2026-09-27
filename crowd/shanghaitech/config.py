"""
Experiment configuration for ShanghaiTech SCDR validation.
"""
from pathlib import Path

# --- Paths (relative to repo root) ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = _REPO_ROOT / "data" / "ShanghaiTech"
OUTPUT_DIR = _REPO_ROOT / "outputs"

# --- Dataset ---
PART = "part_A_final"  # Primary experiment on Part A (dense crowds)
CROP_SIZE = 256        # Random crop size for training
PATCH_SIZE = 32        # Patch size for CSI evaluation

# --- Density map decomposition (threshold-agnostic) ---
# sigma for Gaussian smoothing to produce y_bg = G_sigma * y
# Should match the kernel used to generate density maps from annotations
DECOMP_SIGMA = 15.0  # Part A uses adaptive kernels; this is a smoothing scale

# --- Training ---
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 25
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 5  # Linear LR warmup from 0 to LR over first N epochs

# --- Model ---
ENCODER_BASE_CHANNELS = 64
INPUT_CHANNELS = 3  # RGB

# --- DualDecoder loss weights ---
ALPHA = 0.5   # Background decoder loss weight
BETA_LOSS = 1.5   # Extreme decoder loss weight
DELTA = 0.4   # Final fused prediction loss weight

# --- Evaluation thresholds (persons per PATCH_SIZE x PATCH_SIZE patch) ---
# Part A: tau = 3, 5, 10 gives R ~ 0.09, 0.06, 0.015
THRESHOLDS_A = [3.0, 5.0, 10.0]
# Part B: lower thresholds needed
THRESHOLDS_B = [1.0, 2.0, 3.0]

# --- DualDecoder inference beta sweep ---
DualDecoder_BETAS = [0.8, 1.0, 1.2, 2.0]

# --- R stratification tiers ---
R_LOW = 0.05
R_HIGH = 0.15

"""
Central configuration.

All paths are overridable via environment variables so the project can run
both on Kaggle (original notebook) and locally / in CI:

    export DATA_BASE=/path/to/vietnamese-traffic-signs/archive
    export WORK_DIR=/path/to/working/dir
"""
import os

# Root of the dataset (contains classes.txt, images/, labels/, split_dataset/)
BASE = os.environ.get(
    "DATA_BASE",
    "/kaggle/input/datasets/maitam/vietnamese-traffic-signs/archive",
)

# Scratch/output directory for generated files (YOLO lists, plots, checkpoints)
WORK = os.environ.get("WORK_DIR", "/kaggle/working")

CROP_SIZE = 64
N_CLASSES_TOTAL = 52

WANDB_PROJECT_CLASSIFICATION = "vn-traffic-sign-classification"
WANDB_PROJECT_DETECTION = "vn-traffic-sign-detection"

"""
config.py
---------
Central configuration for the Animal Classification pipeline.
Edit this file to change dataset paths, hyperparameters, and model settings.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR        = "Animal Dataset"
AUGMENTED_DIR      = "Animal Dataset Augmented"
FEATURES_CACHE     = "features_cache.npz"          # cached extracted features
MODEL_SAVE_PATH    = "logistic_regression_model.pkl"
RESULTS_DIR        = "results"                       # plots / metrics saved here

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = ["cat", "cow", "dog", "lamb", "zebra"]

# ── Image Settings ────────────────────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)                         # required by MobileNetV2
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── Augmentation ──────────────────────────────────────────────────────────────
AUGMENT_PER_IMAGE = 3                                # extra images per original
RANDOM_SEED       = 42

# ── Feature Extraction ────────────────────────────────────────────────────────
BATCH_SIZE      = 32
USE_CACHE       = True                               # skip extraction if cache exists

# ── Logistic Regression ───────────────────────────────────────────────────────
LR_PARAMS = {
    "C"           : 0.5,
    "solver"      : "saga",
    "max_iter"    : 3000,
    "random_state": RANDOM_SEED,
}

# ── Train / Test Split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Results ───────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

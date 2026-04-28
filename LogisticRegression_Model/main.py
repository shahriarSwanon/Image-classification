"""
main.py
-------
Full end-to-end pipeline runner.
Runs augmentation → preprocessing → feature extraction → training → evaluation
in one command.

Usage:
    python main.py                   # uses original dataset
    python main.py --augment         # runs augmentation first, then trains
    python main.py --no-cache        # forces re-extraction of features
"""

import argparse
import sys
from pathlib import Path

import config
from augmentation import augment_dataset
from preprocessing import get_dataloader, validate_dataset
from feature_extraction import extract_features
from train import train_model
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Animal Classification Pipeline")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Run offline augmentation before training.",
    )
    parser.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help="Force re-extraction of features (ignore cache).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("   ANIMAL CLASSIFIER  |  Logistic Regression + MobileNetV2")
    print("=" * 60)

    # ── Step 1: Augmentation (optional) ──────────────────────────────────────
    if args.augment:
        print("\n[Step 1/4] Running data augmentation …")
        augment_dataset()
    else:
        print("\n[Step 1/4] Augmentation skipped (use --augment to enable).")

    # ── Step 2: Preprocessing / Dataset Validation ────────────────────────────
    print("\n[Step 2/4] Preprocessing & dataset validation …")
    dataset_dir = (
        config.AUGMENTED_DIR
        if Path(config.AUGMENTED_DIR).exists()
        else config.DATASET_DIR
    )
    print(f"           Dataset: {dataset_dir}")

    if not validate_dataset(dataset_dir):
        print("\nAborting – fix the dataset issues above and retry.")
        sys.exit(1)

    loader, class_names = get_dataloader(dataset_dir)

    # ── Step 3: Feature Extraction ────────────────────────────────────────────
    print("\n[Step 3/4] Extracting deep features (MobileNetV2) …")
    use_cache = not args.no_cache
    X, y = extract_features(loader, class_names, use_cache=use_cache)

    # ── Step 4: Train + Evaluate ──────────────────────────────────────────────
    print("\n[Step 4/4] Training Logistic Regression + evaluation …")
    clf, scaler, X_train, X_test, y_train, y_test = train_model(X, y)

    metrics = evaluate(clf, scaler, X_test, y_test, class_names)

    # ── Summary ───────────────────────────────────────────────────────────────
    acc = metrics["accuracy"]
    print("\n" + "=" * 60)
    if acc >= 0.90:
        print(f"  ✓  Pipeline complete!  Accuracy = {acc:.2%}  (target: 90%+)")
    else:
        print(f"  ✗  Pipeline complete.  Accuracy = {acc:.2%}  (below 90%)")
        print("     Try running with --augment to add more training data.")
    print(f"     Results saved to: {config.RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

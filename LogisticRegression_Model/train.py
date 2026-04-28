"""
train.py
--------
Trains a Logistic Regression model on the pre-extracted MobileNetV2 features,
saves the trained model to disk, and returns the train/test splits for evaluation.

Usage:
    python train.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from preprocessing import get_dataloader, validate_dataset
from feature_extraction import extract_features


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    lr_params: dict  = config.LR_PARAMS,
    seed: int        = config.RANDOM_SEED,
) -> tuple:
    """
    Splits data, scales features, fits Logistic Regression.

    Returns
    -------
    clf     : fitted LogisticRegression
    scaler  : fitted StandardScaler
    X_train, X_test, y_train, y_test : split arrays (already scaled)
    """
    # ── Train / Test Split ────────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    print(f"[Train] Train samples : {len(X_train_raw)}")
    print(f"[Train] Test  samples : {len(X_test_raw)}")

    # ── Feature Scaling ───────────────────────────────────────────────────────
    # StandardScaler substantially improves LR convergence on deep features
    print("[Train] Fitting StandardScaler …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # ── Logistic Regression ───────────────────────────────────────────────────
    # 'multi_class' was removed in scikit-learn 1.7+; multinomial is default
    # when solver='saga' is used with >2 classes.
    print("[Train] Training Logistic Regression …")
    clf = LogisticRegression(**lr_params)
    clf.fit(X_train, y_train)

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_payload = {"model": clf, "scaler": scaler}
    with open(config.MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(save_payload, f)
    print(f"[Train] Model + scaler saved -> {config.MODEL_SAVE_PATH}")

    return clf, scaler, X_train, X_test, y_train, y_test


def load_model(path: str = config.MODEL_SAVE_PATH) -> tuple:
    """
    Loads a previously saved model + scaler pair.

    Returns
    -------
    clf    : LogisticRegression
    scaler : StandardScaler
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["scaler"]


if __name__ == "__main__":
    print("=" * 50)
    print("  MODEL TRAINING (standalone run)")
    print("=" * 50)

    # Pick dataset directory: use augmented if it exists
    dataset_dir = (
        config.AUGMENTED_DIR
        if Path(config.AUGMENTED_DIR).exists()
        else config.DATASET_DIR
    )
    print(f"[Train] Using dataset: {dataset_dir}")

    if not validate_dataset(dataset_dir):
        raise SystemExit(1)

    loader, classes = get_dataloader(dataset_dir)
    X, y            = extract_features(loader, classes)

    clf, scaler, X_train, X_test, y_train, y_test = train_model(X, y)

    # Quick accuracy check
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc  = accuracy_score(y_test,  clf.predict(X_test))
    print(f"\n[Train] Train accuracy : {train_acc:.2%}")
    print(f"[Train] Test  accuracy : {test_acc:.2%}")

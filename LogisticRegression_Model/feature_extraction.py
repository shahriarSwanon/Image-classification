"""
feature_extraction.py
---------------------
Uses a pretrained MobileNetV2 to extract high-dimensional feature vectors
from every image in the dataset.  Results are cached to disk so you don't
have to re-run the slow GPU/CPU forward pass every time you retrain.

Public API:
    extract_features(dataloader, class_names) -> (X, y)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
from tqdm import tqdm

import config


# ── Model Setup ───────────────────────────────────────────────────────────────

def _build_extractor() -> torch.nn.Module:
    """
    Loads pretrained MobileNetV2, strips the classifier head, and sets to eval.
    Works with both old (pretrained=True) and new (weights=...) torchvision APIs.
    """
    print("[FeatureExtraction] Loading pretrained MobileNetV2 …")
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model   = models.mobilenet_v2(weights=weights)
    except AttributeError:                      # older torchvision
        model = models.mobilenet_v2(pretrained=True)

    # Replace classifier with Identity so the model outputs 1280-d features
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


# ── Cache Helpers ─────────────────────────────────────────────────────────────

def _save_cache(X: np.ndarray, y: np.ndarray, class_names: list, path: str) -> None:
    np.savez(path, X=X, y=y, class_names=class_names)
    print(f"[FeatureExtraction] Cache saved -> {path}")


def _load_cache(path: str) -> tuple[np.ndarray, np.ndarray, list] | None:
    p = Path(path)
    if not p.exists():
        return None
    data        = np.load(path, allow_pickle=True)
    class_names = list(data["class_names"])
    print(f"[FeatureExtraction] Cache loaded <- {path}  "
          f"(shape: {data['X'].shape})")
    return data["X"], data["y"], class_names


# ── Main Extraction ───────────────────────────────────────────────────────────

def extract_features(
    dataloader,
    class_names: list,
    cache_path: str  = config.FEATURES_CACHE,
    use_cache: bool  = config.USE_CACHE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 1280-dimensional MobileNetV2 features for every image.

    Parameters
    ----------
    dataloader  : torch DataLoader (from preprocessing.get_dataloader)
    class_names : list[str]  – class labels
    cache_path  : where to load/save the .npz cache
    use_cache   : if True and cache exists, skip extraction entirely

    Returns
    -------
    X : float32 array (N, 1280)
    y : int array   (N,)
    """
    # ── Try cache first ───────────────────────────────────────────────────────
    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            X, y, _ = cached
            return X, y

    # ── Extract fresh features ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FeatureExtraction] Device: {device}")

    model = _build_extractor().to(device)

    all_features, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(
            dataloader,
            desc="[FeatureExtraction] Extracting",
            file=sys.stdout,
        ):
            imgs    = imgs.to(device)
            out     = model(imgs)
            out     = out.view(out.size(0), -1)         # flatten adaptive pool output
            all_features.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features).astype(np.float32)
    y = np.concatenate(all_labels)

    print(f"[FeatureExtraction] Feature matrix: {X.shape}")

    # ── Persist cache ─────────────────────────────────────────────────────────
    _save_cache(X, y, class_names, cache_path)

    return X, y


if __name__ == "__main__":
    from preprocessing import get_dataloader, validate_dataset

    print("=" * 50)
    print("  FEATURE EXTRACTION (standalone run)")
    print("=" * 50)

    if not validate_dataset():
        raise SystemExit(1)

    loader, classes = get_dataloader()
    X, y = extract_features(loader, classes, use_cache=False)

    print(f"\nFeatures : {X.shape}")
    print(f"Labels   : {y.shape}  |  unique={set(y)}")

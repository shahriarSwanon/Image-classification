"""
augmentation.py
---------------
Applies offline data augmentation to balance and expand the training dataset.
Run this ONCE before training if you want more training samples.

Usage:
    python augmentation.py
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm

import config


def random_augment(img: Image.Image) -> Image.Image:
    """Apply a random combination of transforms to a PIL image."""
    ops = []

    # ── Flipping ─────────────────────────────────────────────────────────────
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # ── Rotation (±25°) ───────────────────────────────────────────────────────
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    # ── Brightness (0.7 – 1.3) ───────────────────────────────────────────────
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # ── Contrast (0.7 – 1.3) ─────────────────────────────────────────────────
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # ── Saturation (0.7 – 1.3) ───────────────────────────────────────────────
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Color(img).enhance(factor)

    # ── Sharpness (0.5 – 1.5) ────────────────────────────────────────────────
    if random.random() > 0.5:
        factor = random.uniform(0.5, 1.5)
        img = ImageEnhance.Sharpness(img).enhance(factor)

    # ── Slight blur ──────────────────────────────────────────────────────────
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img


def augment_dataset(
    src_dir: str = config.DATASET_DIR,
    dst_dir: str = config.AUGMENTED_DIR,
    augments_per_image: int = config.AUGMENT_PER_IMAGE,
    seed: int = config.RANDOM_SEED,
) -> None:
    """
    Copies all original images and adds `augments_per_image` synthetic variants
    per image into `dst_dir/<class_name>/`.
    """
    random.seed(seed)
    src_root = Path(src_dir)
    dst_root = Path(dst_dir)

    if dst_root.exists():
        print(f"[Augmentation] Removing existing '{dst_dir}' to start fresh.")
        shutil.rmtree(dst_root)

    original_count = 0
    augmented_count = 0

    print(f"[Augmentation] Source  : {src_dir}")
    print(f"[Augmentation] Output  : {dst_dir}")
    print(f"[Augmentation] Augments per image: {augments_per_image}")
    print()

    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        out_class = dst_root / class_name
        out_class.mkdir(parents=True, exist_ok=True)

        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        print(f"  Class '{class_name}': {len(image_files)} originals -> "
              f"{len(image_files) * (1 + augments_per_image)} total")

        for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    [WARN] Skipping {img_path.name}: {e}")
                continue

            # Copy original
            shutil.copy2(img_path, out_class / img_path.name)
            original_count += 1

            # Write augmented variants
            for i in range(augments_per_image):
                aug_img = random_augment(img)
                stem = img_path.stem
                aug_name = f"{stem}_aug{i+1}.jpg"
                aug_img.save(out_class / aug_name, "JPEG", quality=92)
                augmented_count += 1

    total = original_count + augmented_count
    print(f"\n[Augmentation] Done!  {original_count} originals + "
          f"{augmented_count} augmented = {total} total images.")
    print(f"[Augmentation] Saved to: {dst_dir}")


if __name__ == "__main__":
    augment_dataset()

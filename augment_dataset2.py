#!/usr/bin/env python3
"""
Utility to add simple augmentations (rotations/flips) to dataset2_yolo's train split.

The script assumes YOLOv8 layout created earlier:
dataset2_yolo/
  images/train/*.jpg
  labels/train/*.txt
Existing augmented files containing '_aug' in the stem are skipped so the process
can be re-run without duplicating data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from PIL import Image

TRAIN_IMAGES = Path("dataset2_yolo/images/train")
TRAIN_LABELS = Path("dataset2_yolo/labels/train")

# Maintain label text so YOLO annotations stay in sync after transformations.
def copy_label(stem: str, out_stem: str) -> None:
    src = TRAIN_LABELS / f"{stem}.txt"
    dst = TRAIN_LABELS / f"{out_stem}.txt"
    dst.write_text(src.read_text() if src.exists() else "")


def augment_image(img_path: Path, operations: Dict[str, Callable[[Image.Image], Image.Image]]) -> None:
    stem = img_path.stem
    label_stem = stem
    with Image.open(img_path) as img:
        for suffix, op in operations.items():
            aug = op(img)
            out_stem = f"{stem}_aug_{suffix}"
            aug_path = img_path.with_name(f"{out_stem}{img_path.suffix}")
            aug.save(aug_path)
            copy_label(label_stem, out_stem)


def main() -> None:
    if not TRAIN_IMAGES.exists():
        raise SystemExit(f"Missing train image directory: {TRAIN_IMAGES}")
    operations: Dict[str, Callable[[Image.Image], Image.Image]] = {
        "rot90": lambda im: im.rotate(90, expand=True),
        "rot180": lambda im: im.rotate(180, expand=True),
        "flip_lr": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    }
    image_files = sorted(p for p in TRAIN_IMAGES.iterdir() if p.is_file() and "_aug" not in p.stem)
    if not image_files:
        print("No base images to augment (all contain '_aug').")
        return
    for img_path in image_files:
        augment_image(img_path, operations)
    print(f"Augmented {len(image_files)} images x {len(operations)} operations -> "
          f"{len(image_files) * len(operations)} new samples.")


if __name__ == "__main__":
    main()

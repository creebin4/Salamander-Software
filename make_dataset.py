from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from yolo_utils import YoloPoseLabel, read_yaml, write_yaml, read_label_file, write_label_file
from zoom_augmentations import create_zoom_augmentations


# -----------------------------
# IO helpers
# -----------------------------


def reorder_keypoints(labels: List[YoloPoseLabel]) -> List[YoloPoseLabel]:
    """
    Reorder keypoints according to the specified mapping:
    keypoint 3 -> keypoint 0
    keypoint 4 -> keypoint 1  
    keypoint 2 -> keypoint 2
    keypoint 1 -> keypoint 3
    keypoint 5 -> keypoint 4
    keypoint 0 -> keypoint 5
    """
    reordered_labels = []
    for label in labels:
        if len(label.keypoints) >= 6:  # Ensure we have enough keypoints
            # Reorder keypoints: 3->0, 4->1, 2->2, 1->3, 5->4, 0->5
            reordered_keypoints = [
                label.keypoints[3],  # keypoint 3 -> keypoint 0
                label.keypoints[4],  # keypoint 4 -> keypoint 1
                label.keypoints[2],  # keypoint 5 -> keypoint 4
                label.keypoints[0],  # keypoint 1 -> keypoint 3
                label.keypoints[5],  # keypoint 2 -> keypoint 2
                label.keypoints[1],  # keypoint 0 -> keypoint 5
            ]
            reordered_label = YoloPoseLabel(
                label.class_id,
                label.x_center,
                label.y_center,
                label.width,
                label.height,
                reordered_keypoints
            )
            reordered_labels.append(reordered_label)
        else:
            # If not enough keypoints, keep original
            reordered_labels.append(label)
    return reordered_labels


def list_image_paths(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in image_dir.iterdir() if p.suffix.lower() in exts]


def pair_images_with_labels(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img_path in list_image_paths(image_dir):
        lbl_path = label_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs





# -----------------------------
# Split logic
# -----------------------------


def split_pairs(
    pairs: List[Tuple[Path, Path]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[Tuple[Path, Path]]]:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError("train+val+test ratios must sum to 1.0")
    rng = random.Random(seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)

    n = len(pairs_shuffled)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # Ensure exact sum equals n
    n_test = n - n_train - n_val

    train_split = pairs_shuffled[:n_train]
    val_split = pairs_shuffled[n_train : n_train + n_val]
    test_split = pairs_shuffled[n_train + n_val :]
    return {"train": train_split, "val": val_split, "test": test_split}


# -----------------------------
# Main pipeline
# -----------------------------


def create_splits_and_augment(
    base_data_dir: Path,
    out_base_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    original_images_dir: Path,
    enable_zoom_augmentations: bool = True,
) -> None:
    print("Starting dataset creation with keypoint reordering...")
    print("Keypoint mapping: 3->0, 4->1, 2->2, 1->3, 5->4, 0->5")
    
    src_images = base_data_dir / "train" / "images"
    src_labels = base_data_dir / "train" / "labels"
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(f"Expected {src_images} and {src_labels} to exist with your original data")

    # Read dataset YAML for keypoint config and flip indices
    data_yaml_path = base_data_dir / "data.yaml"
    data_cfg = read_yaml(data_yaml_path)
    kpt_shape = data_cfg.get("kpt_shape", [0, 0])
    num_keypoints = int(kpt_shape[0]) if kpt_shape else 0
    flip_idx = data_cfg.get("flip_idx")
    if isinstance(flip_idx, list) and len(flip_idx) != num_keypoints:
        # Ignore invalid mapping
        flip_idx = None

    pairs = pair_images_with_labels(src_images, src_labels)
    if not pairs:
        raise RuntimeError("No (image,label) pairs found in data/train. Ensure labels exist with matching names.")

    splits = split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed)

    # Prepare output dirs
    for split in ("train", "valid", "test"):
        (out_base_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_base_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy originals for each split
    def copy_pair(img_path: Path, lbl_path: Path, split: str) -> None:
        dst_img = out_base_dir / split / "images" / img_path.name
        dst_lbl = out_base_dir / split / "labels" / lbl_path.name
        shutil.copy2(img_path, dst_img)
        
        # Read and reorder keypoints in labels
        labels = read_label_file(lbl_path)
        original_keypoint_count = sum(len(label.keypoints) for label in labels)
        reordered_labels = reorder_keypoints(labels)
        reordered_keypoint_count = sum(len(label.keypoints) for label in reordered_labels)
        write_label_file(dst_lbl, reordered_labels)
        
        # Log keypoint reordering if any labels had keypoints
        if original_keypoint_count > 0:
            print(f"  Reordered keypoints for {img_path.name}: {original_keypoint_count} -> {reordered_keypoint_count} keypoints")

    for img_path, lbl_path in splits["train"]:
        copy_pair(img_path, lbl_path, "train")
    for img_path, lbl_path in splits["val"]:
        copy_pair(img_path, lbl_path, "valid")
    for img_path, lbl_path in splits["test"]:
        copy_pair(img_path, lbl_path, "test")

    # Create zoom augmentations if enabled
    zoom_augmentation_count = 0
    if enable_zoom_augmentations and original_images_dir.exists():
        print("Creating zoom augmentations from original images...")

        # Get training pairs from output directory
        train_images_dir = out_base_dir / "train" / "images"
        train_labels_dir = out_base_dir / "train" / "labels"
        train_pairs = pair_images_with_labels(train_images_dir, train_labels_dir)
        # Exclude already-augmented images to avoid re-zooming (e.g., *_zoomX.jpg)
        train_pairs = [
            (ip, lp) for (ip, lp) in train_pairs
            if "_zoom" not in ip.stem
        ]

        # Create zoom augmentations with a simple progress indicator
        # Each image gets one zoom with a random factor in [3,4]
        total = len(train_pairs)
        done = 0
        def _progress(msg: str = "") -> None:
            pct = (done / total * 100.0) if total > 0 else 100.0
            print(f"\rAugmenting: {done}/{total} ({pct:5.1f}%) {msg}", end="", flush=True)

        zoom_pairs: List[Tuple[Path, Path]] = []
        for i, (img_path, lbl_path) in enumerate(train_pairs):
            # For each image, call zoom once per factor using the underlying utility
            try:
                # Reuse the internal function on a per-image basis
                single_pairs = create_zoom_augmentations(
                    train_pairs=[(img_path, lbl_path)],
                    out_images_dir=train_images_dir,
                    out_labels_dir=train_labels_dir,
                    original_images_dir=original_images_dir,
                    zoom_factor=2.0 + random.random() * 2.0
                )
                zoom_pairs.extend(single_pairs)
            finally:
                done += 1
                _progress(img_path.name)

        # Finish line
        print()
        zoom_augmentation_count = len(zoom_pairs)
        print(f"Created {zoom_augmentation_count} zoom augmentations")

    # Update data.yaml to point to new splits while preserving other keys
    data_cfg["train"] = str(Path("./train/images"))
    data_cfg["val"] = str(Path("./valid/images"))
    data_cfg["test"] = str(Path("./test/images"))
    
    # Note: flip_idx may need manual adjustment since keypoints were reordered
    if flip_idx:
        print(f"  Note: flip_idx in data.yaml may need manual adjustment due to keypoint reordering")
        print(f"  Original flip_idx: {flip_idx}")
    
    # Ensure required fields are present
    if "nc" not in data_cfg:
        data_cfg["nc"] = 1
    if "names" not in data_cfg:
        data_cfg["names"] = ["object"]

    write_yaml(out_base_dir / "data.yaml", data_cfg)

    # Summary
    total_train_images = len(splits["train"]) + zoom_augmentation_count
    print(json.dumps(
        {
            "counts": {
                "train": total_train_images,
                "val": len(splits["val"]),
                "test": len(splits["test"])
            },
            "original_counts": {k: len(v) for k, v in splits.items()},
            "zoom_augmentations": zoom_augmentation_count,
            "output": str(out_base_dir.resolve()),
            "augmentations": {
                "zoom": enable_zoom_augmentations and zoom_augmentation_count > 0
            },
            "keypoint_reordering": {
                "enabled": True,
                "mapping": "3->0, 4->1, 2->2, 1->3, 5->4, 0->5"
            },
        },
        indent=2,
    ))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split YOLOv8 pose dataset and augment training split")
    p.add_argument("--data-dir", type=str, default="data", help="Base data directory containing train/images and train/labels")
    p.add_argument("--out-dir", type=str, default="new-data", help="Output base directory (will contain train/valid/test)")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio")
    p.add_argument("--val", type=float, default=0.10, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.10, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--disable-zoom", action="store_true", help="Disable zoom augmentations from original images (enabled by default)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    base = Path(args.data_dir)
    out = Path(args.out_dir)

    # Zoom augmentations are enabled by default, disabled with --disable-zoom-augmentations
    enable_zoom = not args.disable_zoom

    # Original images directory is fixed as "images"
    original_images_dir = Path("images") if enable_zoom else None

    create_splits_and_augment(
        base_data_dir=base,
        out_base_dir=out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        original_images_dir=original_images_dir,
        enable_zoom_augmentations=enable_zoom,
    )


if __name__ == "__main__":
    main()

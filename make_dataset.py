from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class YoloPoseLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    # keypoints: list of (x, y, v)
    keypoints: List[Tuple[float, float, float]]

    @staticmethod
    def parse(line: str) -> "YoloPoseLabel":
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 5:
            raise ValueError("Invalid label line; expected at least 5 values")
        class_id = int(parts[0])
        x, y, w, h = parts[1:5]
        remaining = parts[5:]
        keypoints: List[Tuple[float, float, float]] = []
        for i in range(0, len(remaining), 3):
            if i + 2 >= len(remaining):
                break
            keypoints.append((remaining[i], remaining[i + 1], remaining[i + 2]))
        return YoloPoseLabel(class_id, x, y, w, h, keypoints)

    def to_line(self) -> str:
        parts: List[str] = []
        # class id as integer
        parts.append(str(int(self.class_id)))
        # bbox floats
        parts.extend([f"{self.x_center:.6f}", f"{self.y_center:.6f}", f"{self.width:.6f}", f"{self.height:.6f}"])
        # keypoints
        for (kx, ky, v) in self.keypoints:
            parts.append(f"{kx:.6f}")
            parts.append(f"{ky:.6f}")
            # v is typically 0/1/2, keep as int when possible
            v_int = int(round(v))
            parts.append(str(v_int))
        return " ".join(parts)


# -----------------------------
# Label transforms
# -----------------------------


def hflip_label(label: YoloPoseLabel, img_w: int, img_h: int, flip_idx: Sequence[int] | None) -> YoloPoseLabel:
    # Horizontal flip normalized coords: x' = 1 - x
    flipped_kps = [(1.0 - kx, ky, v) for (kx, ky, v) in label.keypoints]
    if flip_idx is not None and len(flip_idx) == len(flipped_kps):
        # Reindex according to mapping; new[i] = old[flip_idx[i]]
        flipped_kps = [flipped_kps[j] for j in flip_idx]

    return YoloPoseLabel(
        class_id=label.class_id,
        x_center=1.0 - label.x_center,
        y_center=label.y_center,
        width=label.width,
        height=label.height,
        keypoints=flipped_kps,
    )


def vflip_label(label: YoloPoseLabel, img_w: int, img_h: int) -> YoloPoseLabel:
    # Vertical flip normalized coords: y' = 1 - y
    flipped_kps = [(kx, 1.0 - ky, v) for (kx, ky, v) in label.keypoints]
    return YoloPoseLabel(
        class_id=label.class_id,
        x_center=label.x_center,
        y_center=1.0 - label.y_center,
        width=label.width,
        height=label.height,
        keypoints=flipped_kps,
    )


# -----------------------------
# IO helpers
# -----------------------------


def read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_label_file(path: Path) -> List[YoloPoseLabel]:
    if not path.exists():
        return []
    labels: List[YoloPoseLabel] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(YoloPoseLabel.parse(line))
    return labels


def write_label_file(path: Path, labels: List[YoloPoseLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(label.to_line() + "\n")


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
# Augment image + labels
# -----------------------------


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # plt.imsave expects RGB in [0,1] or [0,255]; it will handle dtype
    plt.imsave(str(path), array)


def augment_hflip(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img)


def augment_vflip(img: np.ndarray) -> np.ndarray:
    return np.flipud(img)


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
) -> None:
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
        shutil.copy2(lbl_path, dst_lbl)

    for img_path, lbl_path in splits["train"]:
        copy_pair(img_path, lbl_path, "train")
    for img_path, lbl_path in splits["val"]:
        copy_pair(img_path, lbl_path, "valid")
    for img_path, lbl_path in splits["test"]:
        copy_pair(img_path, lbl_path, "test")

    # for img_path, lbl_path in splits["train"]:
    #     img = mpimg.imread(str(img_path))
    #     labels = read_label_file(lbl_path)
    #     if not labels:
    #         continue

    #     stem = img_path.stem
    #     suffix = img_path.suffix
    #     img_w = img.shape[1]
    #     img_h = img.shape[0]

    #     img_hf = augment_hflip(img)
    #     out_img_hf = out_base_dir / "train" / "images" / f"{stem}_hflip{suffix}"
    #     save_image(out_img_hf, img_hf)
    #     labels_hf = [hflip_label(l, img_w, img_h, flip_idx) for l in labels]
    #     out_lbl_hf = out_base_dir / "train" / "labels" / f"{stem}_hflip.txt"
    #     write_label_file(out_lbl_hf, labels_hf)


    #     img_vf = augment_vflip(img)
    #     out_img_vf = out_base_dir / "train" / "images" / f"{stem}_vflip{suffix}"
    #     save_image(out_img_vf, img_vf)
    #     labels_vf = [vflip_label(l, img_w, img_h) for l in labels]
    #     out_lbl_vf = out_base_dir / "train" / "labels" / f"{stem}_vflip.txt"
    #     write_label_file(out_lbl_vf, labels_vf)

    # Update data.yaml to point to new splits while preserving other keys
    data_cfg["train"] = str(Path("./train/images"))
    data_cfg["val"] = str(Path("./valid/images"))
    data_cfg["test"] = str(Path("./test/images"))
    # Ensure required fields are present
    if "nc" not in data_cfg:
        data_cfg["nc"] = 1
    if "names" not in data_cfg:
        data_cfg["names"] = ["object"]

    write_yaml(out_base_dir / "data.yaml", data_cfg)

    # Summary
    print(json.dumps(
        {
            "counts": {k: len(v) for k, v in splits.items()},
            "output": str(out_base_dir.resolve()),
            "augmentations": {"hflip": True, "vflip": True},
        },
        indent=2,
    ))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split YOLOv8 pose dataset and augment training split")
    p.add_argument("--data-dir", type=str, default="data-scaled", help="Base data directory containing train/images and train/labels")
    p.add_argument("--out-dir", type=str, default="new-data", help="Output base directory (will contain train/valid/test)")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio")
    p.add_argument("--val", type=float, default=0.10, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.10, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    base = Path(args.data_dir)
    out = Path(args.out_dir)
    create_splits_and_augment(
        base_data_dir=base,
        out_base_dir=out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
import numpy as np
from PIL import Image as PILImage
import re


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


def list_image_files(path: Path) -> List[Path]:
    """Return a sorted, de-duplicated list of image files from a file or directory path.

    Deduplication key aims to capture IDs like 20250505_Z818592 from filenames such as
    20250505_Z818592.jpg or 20250505_Z818592_png.rf.<hash>.jpg. When duplicates exist,
    prefer files without RoboFlow/hash suffixes (i.e., without 'rf.' or redundant '_png').
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def make_key(p: Path) -> str:
        name = p.stem  # drop extension
        # Remove common RF/hash suffixes in stems like ..._png.rf.<hash>
        # First, cut at '.rf.' if present
        name = name.split('.rf.', 1)[0]
        # If stem still ends with '_png' (from roboflow export), drop it
        if name.endswith('_png'):
            name = name[:-4]
        # Extract leading ID like 20250505_Z818592 if present
        m = re.match(r'^(\d{8}_[A-Za-z0-9]+)', name)
        return m.group(1) if m else name

    def is_preferred_filename(p: Path) -> bool:
        s = p.name
        return ('rf.' not in s) and not s.endswith('_png' + p.suffix)

    if path.is_file():
        return [path] if path.suffix.lower() in supported_exts else []

    if path.is_dir():
        all_images: List[Path] = []
        for ext in supported_exts:
            all_images.extend(path.glob(f"**/*{ext}"))
            all_images.extend(path.glob(f"**/*{ext.upper()}"))
        # Sort for stable preference; shorter names first to naturally prefer simpler names
        all_images = sorted(all_images, key=lambda p: (len(p.name), str(p).lower()))

        key_to_path: Dict[str, Path] = {}
        for img_path in all_images:
            key = make_key(img_path)
            if key not in key_to_path:
                key_to_path[key] = img_path
            else:
                # Prefer non-RF/simple name if available
                current = key_to_path[key]
                if (not is_preferred_filename(current)) and is_preferred_filename(img_path):
                    key_to_path[key] = img_path

        return sorted(key_to_path.values(), key=lambda p: str(p).lower())

    return []


def save_image(path: Path, image: np.ndarray) -> None:
    """Save a numpy image array to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image
    if img.dtype == np.float32 or img.dtype == np.float64:
        # Assume range 0..1; scale to 0..255
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    PILImage.fromarray(img).save(str(path))

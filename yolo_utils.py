from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml


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

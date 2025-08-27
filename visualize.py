import random
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def load_random_image(image_dir: Path) -> Path:
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in supported_exts]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return random.choice(images)


def read_label_file(label_path: Path) -> List[List[float]]:
    if not label_path.exists():
        return []
    rows: List[List[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                values = [float(x) for x in parts]
            except ValueError:
                continue
            if len(values) >= 5:
                rows.append(values)
    return rows


def yolo_to_bbox_pixels(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_px = xc * img_w
    y_px = yc * img_h
    w_px = w * img_w
    h_px = h * img_h
    x_min = x_px - w_px / 2
    y_min = y_px - h_px / 2
    return x_min, y_min, w_px, h_px


def draw_annotation(
    ax,
    label: List[float],
    img_w: int,
    img_h: int,
    keypoint_names: Optional[List[str]] = None,
) -> None:
    # YOLO pose line format:
    # class x y w h kpt1_x kpt1_y kpt1_v kpt2_x kpt2_y kpt2_v ...
    class_id = int(label[0])
    x, y, w, h = label[1], label[2], label[3], label[4]

    # Draw bounding box
    x_min, y_min, w_px, h_px = yolo_to_bbox_pixels(x, y, w, h, img_w, img_h)
    rect = patches.Rectangle((x_min, y_min), w_px, h_px, linewidth=2, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)

    # Label text (numeric class id shown)
    ax.text(
        max(0, x_min),
        max(0, y_min - 5),
        f"head",
        color="yellow",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    # Draw keypoints and connect visible ones in index order
    remaining = label[5:]
    num_keypoints = len(remaining) // 3
    keypoints = []
    for i in range(num_keypoints):
        kx_n = remaining[3 * i + 0]
        ky_n = remaining[3 * i + 1]
        v = remaining[3 * i + 2]
        kx = kx_n * img_w
        ky = ky_n * img_h
        keypoints.append((kx, ky, v))

    # Scatter visible keypoints
    for idx, (kx, ky, v) in enumerate(keypoints):
        if v > 0:  # treat >0 as visible (1 = labeled but not visible, 2 = visible)
            ax.scatter([kx], [ky], c="red", s=30, zorder=3)
            label_text = str(idx)
            if keypoint_names is not None and idx < len(keypoint_names):
                label_text = keypoint_names[idx]
            ax.text(kx + 2, ky + 2, label_text, color="white", fontsize=8, zorder=4)

    # Connect consecutive visible keypoints with lines (0-1-2-...)
    for i in range(len(keypoints) - 1):
        x1, y1, v1 = keypoints[i]
        x2, y2, v2 = keypoints[i + 1]
        if v1 > 0 and v2 > 0:
            ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=2)


def visualize_random_sample():
    base_dir = Path(__file__).resolve().parent
    image_dir = base_dir / "new-data" / "train" / "images"
    label_dir = base_dir / "new-data" / "train" / "labels"

    image_path = load_random_image(image_dir)
    label_path = label_dir / (image_path.stem + ".txt")

    img = mpimg.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img_h = img.shape[0]
    img_w = img.shape[1]

    labels = read_label_file(label_path)

    # Define keypoint names
    keypoint_names = ["0", "1", "2", "3", "4", "5"]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{image_path.name} | labels: {label_path.name if label_path.exists() else 'MISSING'}")
    ax.axis("off")

    for label in labels:
        print(label)
        draw_annotation(ax, label, img_w, img_h, keypoint_names)

    if not labels:
        ax.text(
            10,
            20,
            "No labels found",
            color="white",
            fontsize=12,
            bbox=dict(facecolor="red", alpha=0.5, pad=3),
        )

    plt.tight_layout()
    plt.show()


def main():
    visualize_random_sample()


if __name__ == "__main__":
    main()

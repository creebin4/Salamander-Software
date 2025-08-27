#!/usr/bin/env python3
"""
Script to visualize YOLO pose labels on images.
Shows bounding boxes and keypoints on the given image and label file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from yolo_utils import YoloPoseLabel, read_label_file


def visualize_image_labels(
    image_path: Path,
    label_path: Path,
    keypoint_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Visualize image with bounding boxes and keypoints from YOLO label file.

    Args:
        image_path: Path to the image file
        label_path: Path to the YOLO label file (.txt)
        keypoint_names: Optional list of keypoint names for labeling
        save_path: Optional path to save the visualization
        figsize: Figure size for the plot
    """
    # Load image
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = mpimg.imread(str(image_path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    img_h, img_w = img.shape[:2]

    # Load labels
    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")
        labels = []
    else:
        labels = read_label_file(label_path)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Display image
    ax.imshow(img)
    ax.set_title(f"Image: {image_path.name}\nLabels: {len(labels)} objects")
    ax.axis('off')

    # Colors for different objects
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Draw each label
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]

        # Convert normalized coordinates to pixel coordinates
        x_center = label.x_center * img_w
        y_center = label.y_center * img_h
        width = label.width * img_w
        height = label.height * img_h

        # Calculate bounding box corners
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)

        # Add class label
        class_text = f"Class {int(label.class_id)}"
        ax.text(
            max(0, x1), max(0, y1 - 10),
            class_text,
            color=color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.7, pad=2)
        )

        # Draw keypoints
        for k, (kx, ky, v) in enumerate(label.keypoints):
            if v > 0:  # Only draw visible keypoints
                px = kx * img_w
                py = ky * img_h

                # Draw keypoint
                ax.scatter([px], [py], c=color, s=50, marker='o', edgecolors='white', linewidth=2, zorder=3)

                # Add keypoint label
                kp_label = str(k)
                if keypoint_names and k < len(keypoint_names):
                    kp_label = keypoint_names[k]

                ax.text(
                    px + 5, py + 5,
                    kp_label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, pad=1)
                )

        # Connect keypoints with lines (assuming they form a skeleton)
        visible_kps = [(kx * img_w, ky * img_h) for kx, ky, v in label.keypoints if v > 0]
        if len(visible_kps) > 1:
            # Connect consecutive keypoints
            for j in range(len(visible_kps) - 1):
                x1, y1 = visible_kps[j]
                x2, y2 = visible_kps[j + 1]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for label visualization."""
    p = argparse.ArgumentParser(description="Visualize YOLO pose labels on images")
    p.add_argument("--image", type=str, required=True, help="Path to image file")
    p.add_argument("--label", type=str, required=True, help="Path to YOLO label file (.txt)")
    p.add_argument("--keypoint-names", type=str, help="Comma-separated keypoint names (e.g., 'nose,left-eye,right-eye')")
    p.add_argument("--save", type=str, help="Path to save visualization (if not provided, will display)")
    p.add_argument("--figsize", type=str, default="12,8", help="Figure size as width,height (default: 12,8)")
    return p


def parse_keypoint_names(arg: str) -> List[str]:
    """Parse comma-separated keypoint names."""
    if not arg:
        return []
    return [name.strip() for name in arg.split(",") if name.strip()]


def parse_figsize(arg: str) -> tuple:
    """Parse figsize string as width,height."""
    try:
        parts = arg.split(",")
        return (float(parts[0]), float(parts[1]))
    except (ValueError, IndexError):
        print(f"Warning: Invalid figsize '{arg}', using default (12,8)")
        return (12, 8)


def main() -> None:
    """Main function."""
    args = build_arg_parser().parse_args()

    # Parse arguments
    image_path = Path(args.image)
    label_path = Path(args.label)
    keypoint_names = parse_keypoint_names(args.keypoint_names) if args.keypoint_names else None
    save_path = Path(args.save) if args.save else None
    figsize = parse_figsize(args.figsize)

    # Validate inputs
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")

    # Visualize
    try:
        visualize_image_labels(
            image_path=image_path,
            label_path=label_path,
            keypoint_names=keypoint_names,
            save_path=save_path,
            figsize=figsize
        )
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")
        raise


if __name__ == "__main__":
    main()

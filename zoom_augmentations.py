#!/usr/bin/env python3
"""
Zoom augmentation utilities for YOLO pose datasets.
Provides functions to create zoom augmentations from original high-resolution images.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def extract_image_identifier(filename: str) -> Optional[str]:
    """
    Extract the image identifier in format yyyymmdd_Z000000 from filename.
    Works with both original filenames (20250513_Z820802.jpg) and processed ones
    (20250513_Z820802_png.rf.xxxxx.jpg)
    """
    # Pattern matches: digits_digits_Z followed by 6 digits, then anything else
    pattern = r'(\d{8}_Z\d{6})'
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def find_original_image(image_id: str, images_dir: Path) -> Optional[Path]:
    """
    Find the original full-scale image in the images directory.
    """
    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad image to square while maintaining center.
    Adds padding to top/bottom or left/right as needed.
    """
    height, width = img.shape[:2]

    if height == width:
        return img

    # Determine padding needed
    if height > width:
        # Image is taller, pad left/right
        pad_size = (height - width) // 2
        pad_left = pad_size
        pad_right = height - width - pad_size
        padding = ((0, 0), (pad_left, pad_right), (0, 0)) if len(img.shape) == 3 else ((0, 0), (pad_left, pad_right))
    else:
        # Image is wider, pad top/bottom
        pad_size = (width - height) // 2
        pad_top = pad_size
        pad_bottom = width - height - pad_size
        padding = ((pad_top, pad_bottom), (0, 0), (0, 0)) if len(img.shape) == 3 else ((pad_top, pad_bottom), (0, 0))

    # Pad with zeros (black padding)
    padded = np.pad(img, padding, mode='constant', constant_values=0)
    return padded


def zoom_crop_augmentation(img: np.ndarray, zoom_factor: float = 4.0, crop_ratio: float = 0.8) -> np.ndarray:
    """
    Create a zoom augmentation by cropping the center portion and resizing back to original size.

    Args:
        img: Input image array
        zoom_factor: How much to zoom in (e.g., 4.0 = 4x zoom)
        crop_ratio: What portion of the image to crop (0.8 = crop 80% of the image)

    Returns:
        Zoomed image with same dimensions as input
    """
    height, width = img.shape[:2]

    # Calculate crop dimensions based on zoom factor
    # Higher zoom factor = smaller crop ratio
    effective_crop_ratio = 1.0 / zoom_factor
    crop_height = int(height * effective_crop_ratio)
    crop_width = int(width * effective_crop_ratio)

    # Ensure minimum crop size
    crop_height = max(crop_height, 32)
    crop_width = max(crop_width, 32)

    # Calculate crop coordinates (center crop)
    y_start = (height - crop_height) // 2
    x_start = (width - crop_width) // 2
    y_end = y_start + crop_height
    x_end = x_start + crop_width

    # Crop the center portion
    cropped = img[y_start:y_end, x_start:x_end]

    # Resize back to original dimensions using scipy
    zoom_factors = (height / crop_height, width / crop_width, 1) if len(cropped.shape) == 3 else (height / crop_height, width / crop_width)
    zoomed = ndimage.zoom(cropped, zoom_factors, order=3)

    # Ensure correct dtype
    zoomed = zoomed.astype(img.dtype)

    return zoomed


def zoom_crop_around_label(
    img: np.ndarray,
    labels: List,
    zoom_factor: float,
    target_index: int = 0,
    margin: float = 1.2,
) -> Tuple[np.ndarray, List, float]:
    """
    Zoom into the region around the selected label while ensuring the label's
    bounding box remains fully visible in the cropped view.

    Args:
        img: Padded square image (H==W)
        labels: List of labels (normalized [0,1]) corresponding to the image
        zoom_factor: Desired zoom factor (e.g., 4.0)
        target_index: Index of the label to center the zoom around
        margin: Multiplier to keep context around the box (>=1.0)

    Returns:
        (zoomed_image, transformed_labels, applied_zoom_factor)
    """
    height, width = img.shape[:2]
    S = width  # square
    if not labels:
        # Fallback to center crop if no labels
        return zoom_crop_augmentation(img, zoom_factor), [], zoom_factor

    idx = max(0, min(target_index, len(labels) - 1))
    box = labels[idx]

    # Convert normalized bbox to pixels
    bx = box.x_center * S
    by = box.y_center * S
    bw = box.width * S
    bh = box.height * S

    # Ensure the entire (margin * bbox) fits into the crop
    required_crop_size = max(bw * margin, bh * margin)
    # Crop size corresponding to requested zoom
    requested_crop_size = S / max(zoom_factor, 1e-6)
    crop_size = max(required_crop_size, requested_crop_size)
    # Applied zoom is derived from final crop size
    applied_zoom = S / crop_size

    # Compute top-left of crop so that bbox center is centered, then clamp to image
    x_start = int(round(bx - crop_size / 2))
    y_start = int(round(by - crop_size / 2))
    x_start = max(0, min(x_start, S - int(crop_size)))
    y_start = max(0, min(y_start, S - int(crop_size)))
    x_end = x_start + int(crop_size)
    y_end = y_start + int(crop_size)

    # Crop and resize back to original size (fast via OpenCV)
    cropped = img[y_start:y_end, x_start:x_end]
    try:
        import cv2  # type: ignore
        interp = cv2.INTER_LINEAR if S >= cropped.shape[0] else cv2.INTER_AREA
        zoomed_img = cv2.resize(cropped, (S, S), interpolation=interp)
    except Exception:
        # Fallback to scipy if OpenCV is unavailable
        from scipy import ndimage as _nd
        zf = (
            S / cropped.shape[0],
            S / cropped.shape[1],
            1,
        ) if len(cropped.shape) == 3 else (
            S / cropped.shape[0],
            S / cropped.shape[1],
        )
        zoomed_img = _nd.zoom(cropped, zf, order=1)
    zoomed_img = zoomed_img.astype(img.dtype)

    # Transform all labels into the zoomed view
    transformed_labels: List = []
    inv_crop_size = 1.0 / crop_size
    for lbl in labels:
        nx = (lbl.x_center * S - x_start) * inv_crop_size
        ny = (lbl.y_center * S - y_start) * inv_crop_size
        nw = (lbl.width * S) * inv_crop_size
        nh = (lbl.height * S) * inv_crop_size

        def _c(val: float) -> float:
            return max(0.0, min(1.0, val))

        new_kps = []
        for kx, ky, v in lbl.keypoints:
            kxp = (kx * S - x_start) * inv_crop_size
            kyp = (ky * S - y_start) * inv_crop_size
            new_kps.append((_c(kxp), _c(kyp), v))

        # Create a proper YoloPoseLabel so write_label_file can serialize via to_line()
        from yolo_utils import YoloPoseLabel  # local import to avoid circulars at module import time
        new_lbl = YoloPoseLabel(
            class_id=int(lbl.class_id),
            x_center=_c(nx),
            y_center=_c(ny),
            width=_c(nw),
            height=_c(nh),
            keypoints=new_kps,
        )
        transformed_labels.append(new_lbl)

    return zoomed_img, transformed_labels, applied_zoom


def save_image(path: Path, array: np.ndarray) -> None:
    """
    Save image array to file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # plt.imsave expects RGB in [0,1] or [0,255]; it will handle dtype
    plt.imsave(str(path), array)


def generate_zoom_for_pair(
    img_path: Path,
    lbl_path: Path,
    original_images_dir: Path,
    zoom_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List, float, str, str]:
    """
    Core zoom pipeline used by both dataset creation and visualization.

    Returns:
        original_img, padded_img, zoomed_img, original_labels, zoomed_labels,
        applied_zoom, out_img_name, out_lbl_name
    """
    from yolo_utils import read_label_file

    # Extract image identifier and find original high-res image
    image_id = extract_image_identifier(img_path.name)
    if not image_id:
        raise RuntimeError(f"Could not extract image ID from {img_path.name}")

    original_img_path = find_original_image(image_id, original_images_dir)
    if not original_img_path:
        raise RuntimeError(f"Could not find original image for {image_id}")

    # Load original image
    original_img = mpimg.imread(str(original_img_path))
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)

    # Load labels
    original_labels = read_label_file(lbl_path)
    if not original_labels:
        raise RuntimeError(f"No labels found for {lbl_path}")

    # Pad original to square
    padded_img = pad_to_square(original_img)

    # Head-centric zoom
    zoomed_img, zoomed_labels, applied_zoom = zoom_crop_around_label(
        padded_img, original_labels, zoom_factor=zoom_factor, target_index=0, margin=1.2
    )

    # Output names
    stem = img_path.stem
    suffix = img_path.suffix
    zoom_suffix = f"_zoom{applied_zoom:.0f}"
    out_img_name = f"{stem}{zoom_suffix}{suffix}"
    out_lbl_name = f"{stem}{zoom_suffix}.txt"

    return (
        original_img,
        padded_img,
        zoomed_img,
        original_labels,
        zoomed_labels,
        applied_zoom,
        out_img_name,
        out_lbl_name,
    )


def create_zoom_augmentations(
    train_pairs: List[Tuple[Path, Path]],
    out_images_dir: Path,
    out_labels_dir: Path,
    original_images_dir: Path,
    zoom_factor: float = 3.5,
) -> List[Tuple[Path, Path]]:
    """
    Create zoom augmentations from original high-resolution images using
    head-centered zoom consistent with the interactive tester.

    Args:
        train_pairs: List of (image_path, label_path) tuples from current training set
        out_images_dir: Directory to save augmented images
        out_labels_dir: Directory to save augmented labels
        original_images_dir: Directory containing original high-resolution images
        zoom_factor: Zoom factor to apply (e.g., 3.5)

    Returns:
        List of (augmented_image_path, augmented_label_path) tuples
    """
    from yolo_utils import YoloPoseLabel, read_label_file, write_label_file

    augmented_pairs = []

    for img_path, lbl_path in train_pairs:
        # Extract image identifier
        image_id = extract_image_identifier(img_path.name)
        if not image_id:
            print(f"Warning: Could not extract image ID from {img_path.name}")
            continue

        # Find original image
        original_img_path = find_original_image(image_id, original_images_dir)
        if not original_img_path:
            print(f"Warning: Could not find original image for {image_id}")
            continue

        try:
            # Load original image
            original_img = mpimg.imread(str(original_img_path))
            if original_img.dtype != np.uint8:
                original_img = (original_img * 255).astype(np.uint8)

            # Load labels
            labels = read_label_file(lbl_path)
            if not labels:
                continue

            # Pad to square
            padded_img = pad_to_square(original_img)
            padded_height, padded_width = padded_img.shape[:2]

            # Get original dimensions for label transformation
            orig_height, orig_width = original_img.shape[:2]

            # Calculate padding offsets
            pad_y = (padded_height - orig_height) // 2
            pad_x = (padded_width - orig_width) // 2

            (
                _orig_img,
                _padded_img,
                zoomed_img,
                _orig_labels,
                transformed_labels,
                applied_zoom,
                out_img_name,
                out_lbl_name,
            ) = generate_zoom_for_pair(
                img_path=img_path,
                lbl_path=lbl_path,
                original_images_dir=original_images_dir,
                zoom_factor=zoom_factor,
            )

            out_img_path = out_images_dir / out_img_name
            out_lbl_path = out_labels_dir / out_lbl_name

            # Save augmented image and labels
            save_image(out_img_path, zoomed_img)
            write_label_file(out_lbl_path, transformed_labels)

            augmented_pairs.append((out_img_path, out_lbl_path))
            print(f"Created zoom augmentation: {out_img_name}")

        except Exception as e:
            print(f"Error processing {original_img_path}: {e}")
            continue

    return augmented_pairs


def visualize_zoom_augmentation(
    original_img: np.ndarray,
    padded_img: np.ndarray,
    zoomed_img: np.ndarray,
    zoom_factor: float,
    padded_labels: List,  # Labels scaled for padded image
    zoomed_labels: List,  # Labels transformed for zoomed image
    keypoint_names: Optional[List[str]] = None,
) -> None:
    """
    Visualize the zoom augmentation process using the shared visualization approach.
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    images_and_labels = [
        (original_img, None, "Original Image"),  # No labels for original (they won't match)
        (padded_img, padded_labels, "Padded to Square"),  # Labels scaled for padded image
        (zoomed_img, zoomed_labels, f"Zoom {zoom_factor}x")
    ]

    # Colors for different objects
    colors = ['lime', 'red', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, (img, labels, title) in enumerate(images_and_labels):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

        # Only draw labels if they exist
        if labels is not None:
            # Draw each label
            for j, label in enumerate(labels):
                color = colors[j % len(colors)]

                # Convert normalized coordinates to pixel coordinates
                x_center = label.x_center * img.shape[1]
                y_center = label.y_center * img.shape[0]
                width = label.width * img.shape[1]
                height = label.height * img.shape[0]

                # Calculate bounding box corners
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                # Draw bounding box
                import matplotlib.patches as patches
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
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2)
                )

                # Draw keypoints
                for k, (kx, ky, v) in enumerate(label.keypoints):
                    if v > 0:  # Only draw visible keypoints
                        px = kx * img.shape[1]
                        py = ky * img.shape[0]

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
                            fontsize=8,
                            fontweight='bold',
                            bbox=dict(facecolor=color, alpha=0.8, pad=1)
                        )

                # Connect keypoints with lines (assuming they form a skeleton)
                visible_kps = [(kx * img.shape[1], ky * img.shape[0]) for kx, ky, v in label.keypoints if v > 0]
                if len(visible_kps) > 1:
                    # Connect consecutive keypoints
                    for m in range(len(visible_kps) - 1):
                        x1, y1 = visible_kps[m]
                        x2, y2 = visible_kps[m + 1]
                        ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.close()


def test_single_zoom(
    train_images_dir: Path = Path("data/train/images"),
    original_images_dir: Path = Path("images"),
    zoom_factor: float = 4.0,
    max_image_size: Optional[int] = 2048,  # Max dimension to prevent memory issues
) -> None:
    """
    Test zoom augmentation on a random image from the training set.
    """
    from yolo_utils import read_label_file
    import cv2

    print("🔍 Starting zoom augmentation test...")

    # Get list of training images (support common formats incl. TIFF)
    image_files: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        image_files.extend(train_images_dir.glob(pattern))
        image_files.extend(train_images_dir.glob(pattern.upper()))
    # Deduplicate and sort
    image_files = sorted(set(image_files))
    if not image_files:
        print(f"❌ No image files found in {train_images_dir}")
        return

    print(f"📁 Found {len(image_files)} images in training directory")

    # Pick random image
    random_image = random.choice(image_files)
    print(f"🎲 Selected random image: {random_image.name}")

    # Extract image identifier
    image_id = extract_image_identifier(random_image.name)
    if not image_id:
        print(f"❌ Could not extract image ID from {random_image.name}")
        return

    print(f"🆔 Extracted image ID: {image_id}")

    # Find original image
    original_img_path = find_original_image(image_id, original_images_dir)
    if not original_img_path:
        print(f"❌ Could not find original image for {image_id} in {original_images_dir}")
        return

    print(f"📸 Found original image: {original_img_path}")

    # Load original image with size check
    print("⏳ Loading original image...")
    try:
        # Try matplotlib first, fallback to cv2
        try:
            original_img = mpimg.imread(str(original_img_path))
        except Exception as e:
            print(f"⚠️  Matplotlib failed, trying OpenCV: {e}")
            original_img = cv2.imread(str(original_img_path), cv2.IMREAD_COLOR)
            if original_img is None:
                raise RuntimeError(f"Failed to load image with OpenCV")
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        if original_img is None or original_img.size == 0:
            print("❌ Failed to load image")
            return

        # Check image size and resize if too large
        height, width = original_img.shape[:2]
        max_dim = max(height, width)
        if max_image_size is not None and max_dim > max_image_size:
            scale_factor = max_image_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"📏 Image too large ({width}x{height}), resizing to {new_width}x{new_height}")

            if original_img.shape[2] == 3:  # RGB
                original_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:  # Grayscale
                original_img = cv2.resize(original_img[:, :, 0], (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                original_img = original_img[:, :, np.newaxis]

        if original_img.dtype != np.uint8:
            original_img = (original_img * 255).astype(np.uint8)

        print(f"✅ Original image loaded: {original_img.shape}")

    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # Load labels
    label_file = random_image.parent.parent / "labels" / f"{random_image.stem}.txt"
    print(f"🏷️  Loading labels from: {label_file}")
    original_labels = read_label_file(label_file)
    print(f"✅ Loaded {len(original_labels)} labels")

    # Pad original high-res image to square
    print("⏳ Padding high-res image to square...")
    padded_img = pad_to_square(original_img)
    padded_height, padded_width = padded_img.shape[:2]
    print(f"✅ Padded image size: {padded_width}x{padded_height}")

    # Labels are normalized [0,1] relative to the square dataset images.
    # Since we also padded to a square, we can reuse the same normalized labels for the padded image.
    padded_labels = original_labels

    # Use the same core function as dataset generation
    print(f"⏳ Applying zoom around head (index 0) with requested {zoom_factor}x...")
    try:
        (
            _orig2,
            _padded2,
            zoomed_img,
            _orig_labels2,
            zoomed_labels,
            applied_zoom,
            _out_img_name,
            _out_lbl_name,
        ) = generate_zoom_for_pair(
            img_path=random_image,
            lbl_path=label_file,
            original_images_dir=original_images_dir,
            zoom_factor=zoom_factor,
        )
        print(f"✅ Zoomed image size: {zoomed_img.shape} (applied zoom ~{applied_zoom:.2f}x)")
    except Exception as e:
        print(f"❌ Error during zoom augmentation: {e}")
        return

    # Note: zoom_crop_around_label already transformed labels

    print("🎨 Creating visualization...")

    # Visualize with error handling
    try:
        visualize_zoom_augmentation(
            original_img, padded_img, zoomed_img, zoom_factor,
            padded_labels=padded_labels,
            zoomed_labels=zoomed_labels
        )
        print("✅ Visualization completed!")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        print("💡 Try running with a smaller image or check matplotlib backend")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for zoom augmentation testing."""
    p = argparse.ArgumentParser(description="Test zoom augmentation on random training image")
    p.add_argument("--train-dir", type=str, default="data/train/images", help="Directory containing training images")
    p.add_argument("--original-dir", type=str, default="images", help="Directory containing original high-resolution images")
    p.add_argument("--zoom-factor", type=float, default=0.0, help="Zoom factor to apply (0 = random in [3,4])")
    p.add_argument("--max-size", type=int, default=2048, help="Maximum image dimension (default: 2048)")
    p.add_argument("--no-resize", action="store_true", help="Don't resize large images (may cause hangs)")
    p.add_argument("--headless", action="store_true", help="Use non-interactive matplotlib backend (for headless environments)")
    return p


def main() -> None:
    """Main function to test zoom augmentation."""
    args = build_arg_parser().parse_args()

    # Set matplotlib backend based on headless flag
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
        print("🔧 Using non-interactive matplotlib backend (headless mode)")

    train_images_dir = Path(args.train_dir)
    original_images_dir = Path(args.original_dir)

    if not train_images_dir.exists():
        print(f"❌ Training images directory does not exist: {train_images_dir}")
        return

    if not original_images_dir.exists():
        print(f"❌ Original images directory does not exist: {original_images_dir}")
        return

    max_image_size = None if args.no_resize else args.max_size

    try:
        # Choose zoom: if 0 or negative, pick random in [3,4]
        zf = args.zoom_factor if args.zoom_factor and args.zoom_factor > 0 else (3.0 + random.random() * 1.0)
        print(f"🔎 Using zoom factor: {zf:.2f}")
        test_single_zoom(
            train_images_dir,
            original_images_dir,
            zf,
            max_image_size
        )
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

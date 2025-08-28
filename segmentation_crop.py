import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from PIL import Image as PILImage

from yolo_utils import read_yaml, list_image_files, save_image


def load_segmentation_model(model_path: Optional[str] = None) -> YOLO:
    """Load YOLO segmentation model"""
    if model_path is None:
        model_path = "yolov8s-seg.pt"  # Default segmentation model
    return YOLO(model_path)


def load_segmentation_models(model_paths: List[str]) -> List[YOLO]:
    """Load multiple YOLO segmentation models, skipping any that fail to load."""
    loaded: List[YOLO] = []
    for path in model_paths:
        try:
            model = YOLO(path)
            # attach source path for later logging
            try:
                setattr(model, "_source_path", path)
            except Exception:
                pass
            loaded.append(model)
        except Exception as e:
            print(f"Warning: failed to load model '{path}': {e}")
            continue
    return loaded


def get_image_paths(path: Path) -> List[Path]:
    """Get all image files from a path (single file or directory)"""
    images = list_image_files(path)
    if not images:
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        else:
            raise ValueError(f"No supported images found in {path}")
    return images


def run_segmentation_inference(
    image_path: Path,
    model: YOLO,
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    List[str],
    List[np.ndarray],
    List[int],
]:
    """
    Run YOLO segmentation inference on a single image.
    
    Args:
        image_path: Path to the input image
        model: Loaded YOLO segmentation model
        imgsz: Inference image size (default: 640)
        conf: Confidence threshold (default: 0.25)
        
    Returns:
        Tuple of (original_image, combined_mask, class_names, instance_masks_resized, class_ids)
        - original_image: Original input image as numpy array
        - combined_mask: Single mask [H, W] with instance indices (1..N), or None
        - class_names: List of detected class names (parallel to instances)
        - instance_masks_resized: List of boolean masks [H, W] per instance
        - class_ids: List of integer class ids per instance
    """
    # Load original image
    orig_img = mpimg.imread(str(image_path))
    if orig_img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    
    # Convert to RGB if needed
    if orig_img.ndim == 2:  # grayscale
        orig_img = np.stack([orig_img] * 3, axis=-1)
    elif orig_img.shape[2] == 4:  # RGBA
        orig_img = orig_img[:, :, :3]  # drop alpha
    
    # Run segmentation inference
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    
    if not results or results[0].masks is None:
        return orig_img, None, [], [], []
    
    result = results[0]
    
    # Get segmentation masks
    masks = result.masks.data.cpu().numpy()  # [N, Hm, Wm]

    instance_masks_resized: List[np.ndarray] = []
    # Combine all masks into a single mask
    if masks.shape[0] > 0:
        # Resize masks to original image size
        height, width = orig_img.shape[0], orig_img.shape[1]
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for i, mask in enumerate(masks):
            # Resize mask to match original image dimensions
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((width, height), PILImage.Resampling.LANCZOS)
            mask_resized_np = (np.array(mask_resized) > 127).astype(np.uint8)

            # keep per-instance boolean mask
            instance_masks_resized.append(mask_resized_np.astype(bool))

            # Add to combined mask with different values for different instances
            combined_mask[mask_resized_np > 0] = i + 1
    else:
        combined_mask = None
    
    # Get class names for detected objects
    detected_classes: List[str] = []
    class_ids_list: List[int] = []
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_ids_list = list(map(int, class_ids.tolist()))
        class_names = result.names if hasattr(result, "names") and isinstance(result.names, dict) else model.names

        if isinstance(class_names, dict):
            detected_classes = [class_names.get(cls_id, f"class_{cls_id}") for cls_id in class_ids_list]
        else:
            detected_classes = [class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}" for cls_id in class_ids_list]

    return orig_img, combined_mask, detected_classes, instance_masks_resized, class_ids_list


def visualize_segmentation_results(
    original_image: np.ndarray,
    segmentation_mask: Optional[np.ndarray],
    class_names: List[str],
    image_name: str,
    cropped_image: Optional[np.ndarray] = None,
    crop_margins: Optional[Tuple[int, int, int]] = None,
) -> None:
    """
    Visualize original image and segmentation results using matplotlib.
    
    Args:
        original_image: Original input image
        segmentation_mask: Segmentation mask or None if no detections
        class_names: List of detected class names
        image_name: Name of the image file for title
    """
    num_cols = 4 if cropped_image is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image\n{image_name}")
    axes[0].axis("off")
    
    # Segmentation mask
    if segmentation_mask is not None:
        axes[1].imshow(segmentation_mask, cmap='tab10', alpha=0.8)
        axes[1].set_title(f"Segmentation Mask\nDetected: {len(class_names)} objects")
        axes[1].axis("off")
        
        # Overlay segmentation on original
        axes[2].imshow(original_image)
        axes[2].imshow(segmentation_mask, cmap='tab10', alpha=0.5)
        axes[2].set_title(f"Original + Segmentation Overlay\nClasses: {', '.join(set(class_names))}")
        axes[2].axis("off")
        
        # Cropped image panel (if provided)
        if cropped_image is not None and num_cols >= 4:
            axes[3].imshow(cropped_image)
            if crop_margins is not None:
                t, b, r = crop_margins
                axes[3].set_title(f"Cropped Image\n(top={t}, bottom={b}, right={r})")
            else:
                axes[3].set_title("Cropped Image")
            axes[3].axis("off")
    else:
        axes[1].text(0.5, 0.5, "No objects detected", ha='center', va='center', fontsize=16, 
                    transform=axes[1].transAxes)
        axes[1].set_title("Segmentation Mask\nNo detections")
        axes[1].axis("off")
        
        axes[2].imshow(original_image)
        axes[2].set_title("Original + Segmentation Overlay\nNo detections")
        axes[2].axis("off")
        
        # Cropped image panel placeholder when no detections
        if cropped_image is not None and num_cols >= 4:
            axes[3].imshow(cropped_image)
            if crop_margins is not None:
                t, b, r = crop_margins
                axes[3].set_title(f"Cropped Image\n(top={t}, bottom={b}, right={r})")
            else:
                axes[3].set_title("Cropped Image")
            axes[3].axis("off")
    
    plt.tight_layout()
    plt.show()


def select_largest_instance_mask(instance_masks: List[np.ndarray]) -> Optional[np.ndarray]:
    """Return the largest instance mask by area. If none, return None."""
    if not instance_masks:
        return None
    if len(instance_masks) == 1:
        return instance_masks[0]
    areas = [int(mask.sum()) for mask in instance_masks]
    largest_index = int(np.argmax(areas))
    return instance_masks[largest_index]


def largest_instance_in_center(instance_masks: List[np.ndarray], image_shape: Tuple[int, int], center_fraction: float = 1.0/3.0) -> bool:
    """
    Return True if the largest instance intersects the central rectangle of the image.
    center_fraction defines the width and height fraction of the central region (default 1/3).
    """
    if not instance_masks:
        return False
    largest = select_largest_instance_mask(instance_masks)
    if largest is None:
        return False
    height, width = image_shape
    cf = float(center_fraction)
    cf = max(0.0, min(1.0, cf))
    cx_start = int(round((1.0 - cf) / 2.0 * width))
    cx_end = int(round((1.0 + cf) / 2.0 * width))
    cy_start = int(round((1.0 - cf) / 2.0 * height))
    cy_end = int(round((1.0 + cf) / 2.0 * height))
    cx_start = max(0, min(cx_start, width - 1))
    cx_end = max(cx_start + 1, min(cx_end, width))
    cy_start = max(0, min(cy_start, height - 1))
    cy_end = max(cy_start + 1, min(cy_end, height))
    center_region = largest[cy_start:cy_end, cx_start:cx_end]
    return bool(np.any(center_region))


def compute_top_crop_using_middle_vertical_third(instance_mask: np.ndarray) -> int:
    """
    Compute how many pixels can be cropped from the top without removing the object,
    using a right-of-center sixth of the image width (narrower and shifted right).
    """
    height, width = instance_mask.shape
    col_start = int((4 * width) / 6)
    col_end = int((5 * width) / 6)
    if col_end <= col_start:
        col_start = max(0, min(col_start, width - 1))
        col_end = min(width, col_start + 1)
    middle_vertical = instance_mask[:, col_start:col_end]
    rows_with_object = np.any(middle_vertical, axis=1)
    if not np.any(rows_with_object):
        return 0
    first_row_with_object = int(np.where(rows_with_object)[0][0])
    return max(first_row_with_object, 0)


def compute_bottom_crop_using_middle_vertical_third(instance_mask: np.ndarray) -> int:
    """
    Compute how many pixels can be cropped from the bottom without removing the object,
    using a right-of-center sixth of the image width (narrower and shifted right).
    """
    height, width = instance_mask.shape
    col_start = int((4 * width) / 6)
    col_end = int((5 * width) / 6)
    if col_end <= col_start:
        col_start = max(0, min(col_start, width - 1))
        col_end = min(width, col_start + 1)
    middle_vertical = instance_mask[:, col_start:col_end]
    rows_with_object = np.any(middle_vertical, axis=1)
    if not np.any(rows_with_object):
        return 0
    last_row_with_object = int(np.where(rows_with_object)[0][-1])
    crop_from_bottom = int((height - 1) - last_row_with_object)
    return max(crop_from_bottom, 0)


def compute_right_crop_using_middle_horizontal_third(instance_mask: np.ndarray) -> int:
    """
    Compute how many pixels can be cropped from the right without removing the object,
    using only the middle horizontal third of the image.
    """
    height, width = instance_mask.shape
    row_start = height // 3
    row_end = (2 * height) // 3
    middle_horizontal = instance_mask[row_start:row_end, :]
    cols_with_object = np.any(middle_horizontal, axis=0)
    if not np.any(cols_with_object):
        return 0
    rightmost_col_with_object = int(np.where(cols_with_object)[0][-1])
    crop_from_right = int((width - 1) - rightmost_col_with_object)
    return max(crop_from_right, 0)


def crop_image_using_margins(image: np.ndarray, top: int, bottom: int, right: int) -> np.ndarray:
    """Return a copy of the image cropped by the specified margins.
    Left margin is kept at 0 per requirements.
    """
    height, width = image.shape[:2]
    top = max(0, int(top))
    bottom = max(0, int(bottom))
    right = max(0, int(right))
    # Hard cap: do not remove more than 1/3 from any side
    max_top = height // 3
    max_bottom = height // 3
    max_right = width // 3
    top = min(top, max_top)
    bottom = min(bottom, max_bottom)
    right = min(right, max_right)
    # Bound the crop values so we don't invert or go out of bounds
    top = min(top, height - 1)
    bottom = min(bottom, height - 1 - top)
    right = min(right, width - 1)
    cropped = image[top: height - bottom, 0: width - right]
    return cropped


def run_segmentation_on_images(
    input_path: str,
    model_path: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.25,
    visualize: bool = False,
    tb_add_back_pct: float = 0.0,
) -> None:
    """
    Main function to run segmentation on images from a folder or single image.
    
    Args:
        input_path: Path to input image or folder containing images
        model_path: Path to YOLO segmentation model (default: yolov8s-seg.pt)
        imgsz: Inference image size (default: 640)
        conf: Confidence threshold (default: 0.25)
    """
    # Resolve models to try: primary + default fallbacks
    primary = model_path if model_path is not None else "yolov8s-seg.pt"
    default_fallbacks: List[str] = ["yolov8x-seg.pt", "yolov9c-seg.pt", "yolov9e-seg.pt", "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11x-seg.pt"]
    models_to_try_paths: List[str] = [primary] + default_fallbacks
    # Deduplicate while preserving order
    seen = set()
    models_to_try_paths = [m for m in models_to_try_paths if not (m in seen or seen.add(m))]

    print(f"Loading YOLO segmentation models: {', '.join(models_to_try_paths)}")
    models_to_try = load_segmentation_models(models_to_try_paths)
    if not models_to_try:
        raise RuntimeError("No valid segmentation models could be loaded.")
    
    # Get list of images to process
    path = Path(input_path)
    image_paths = get_image_paths(path)

    # Prepare output directory 'better-crops'
    output_base = (path / "better-crops") if path.is_dir() else (path.parent / "better-crops")
    output_base.mkdir(parents=True, exist_ok=True)
    
    if not image_paths:
        raise ValueError(f"No images found at {input_path}")
    
    print(f"Found {len(image_paths)} image(s) to process")
    
    total_start_time = time.time()
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            start_time = time.time()
            
            # Try models in order until detections are found
            original_img = None
            seg_mask = None
            class_names: List[str] = []
            instance_masks: List[np.ndarray] = []
            class_ids: List[int] = []
            used_model_name: Optional[str] = None
            for model in models_to_try:
                model_path_str = getattr(model, '_source_path', None)
                model_name_for_log = Path(model_path_str).name if model_path_str else str(model)
                print(f"  - Trying model: {model_name_for_log}")
                img, seg, names, inst_masks, cls_ids = run_segmentation_inference(
                    image_path=image_path,
                    model=model,
                    imgsz=imgsz,
                    conf=conf,
                )
                # Initialize original image once
                if original_img is None:
                    original_img = img
                # Accept model only if detections exist and largest instance intersects center
                if seg is not None and len(inst_masks) > 0 and largest_instance_in_center(inst_masks, img.shape[:2]):
                    print(f"    -> Success with model: {model_name_for_log} (largest instance present in center)")
                    seg_mask = seg
                    class_names = names
                    instance_masks = inst_masks
                    class_ids = cls_ids
                    used_model_name = getattr(model, 'ckpt_path', None) or getattr(model, 'model', None) or str(model)
                    break
                else:
                    if seg is None or len(inst_masks) == 0:
                        print(f"    -> No detections with model: {model_name_for_log}")
                    else:
                        print(f"    -> Rejected model: {model_name_for_log} (largest instance not in center)")
            if used_model_name is None:
                # No detections with any model; keep last original_img
                used_model_name = models_to_try_paths[0]
            
            # Compute crop amounts using the largest instance and middle-third strategies
            largest_mask = select_largest_instance_mask(instance_masks)
            if largest_mask is not None:
                top_crop = compute_top_crop_using_middle_vertical_third(largest_mask)
                bottom_crop = compute_bottom_crop_using_middle_vertical_third(largest_mask)
                right_crop = compute_right_crop_using_middle_horizontal_third(largest_mask)
                h, w = original_img.shape[:2]
                # Optionally add back a percentage of image height to top/bottom crops
                if tb_add_back_pct and tb_add_back_pct > 0:
                    add_back_px = int(round((tb_add_back_pct / 100.0) * h))
                    top_crop = max(0, top_crop - add_back_px)
                    bottom_crop = max(0, bottom_crop - add_back_px)
                # Clamp suggestions to at most 1/3 of dimension for safety
                top_crop = min(top_crop, h // 3)
                bottom_crop = min(bottom_crop, h // 3)
                right_crop = min(right_crop, w // 3)
                print(f"Suggested crops (pixels): top={top_crop}, bottom={bottom_crop}, right={right_crop} (model: {Path(used_model_name).name})")
            else:
                print(f"Suggested crops (pixels): top=0, bottom=0, right=0 (no instances; tried {len(models_to_try)} model(s))")

            # Always crop for saving, even if not visualizing
            if largest_mask is not None:
                cropped_img = crop_image_using_margins(original_img, top_crop, bottom_crop, right_crop)
                crop_margins = (top_crop, bottom_crop, right_crop)
            else:
                cropped_img = original_img
                crop_margins = None

            # Determine output path, preserving structure if input is a directory
            if path.is_dir():
                rel_path = image_path.relative_to(path)
                out_path = output_base / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = output_base / image_path.name

            # Save cropped image
            save_image(out_path, cropped_img)

            # Optionally visualize
            if visualize:
                visualize_segmentation_results(
                    original_image=original_img,
                    segmentation_mask=seg_mask,
                    class_names=class_names,
                    image_name=image_path.name,
                    cropped_image=cropped_img,
                    crop_margins=crop_margins,
                )
            
            processing_time = time.time() - start_time
            print(f"Processing time: {processing_time:.2f}s")
            
            if class_names:
                print(f"Detected classes: {', '.join(set(class_names))}")
            else:
                print("No objects detected")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    print(f"\nFinished processing {len(image_paths)} image(s) in {total_time:.2f}s")


def main():
    """Command line interface for segmentation crop"""
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on images; save crops to 'better-crops'; optional visualization")
    parser.add_argument("input", type=str, help="Path to input image or folder containing images")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to YOLO segmentation model (default: yolov8s-seg.pt)")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.2, 
                       help="Confidence threshold (default: 0.2)")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization panels")
    parser.add_argument("--tb-add-back-pct", type=float, default=5.0, 
                       help="Add back percentage of image height to top/bottom crops (e.g., 5.0)")
    
    args = parser.parse_args()
    
    run_segmentation_on_images(
        input_path=args.input,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        visualize=args.visualize,
        tb_add_back_pct=args.tb_add_back_pct,
    )


if __name__ == "__main__":
    main()

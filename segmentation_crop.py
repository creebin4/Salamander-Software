import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from PIL import Image as PILImage

from yolo_utils import read_yaml


def load_segmentation_model(model_path: Optional[str] = None) -> YOLO:
    """Load YOLO segmentation model"""
    if model_path is None:
        model_path = "yolov8s-seg.pt"  # Default segmentation model
    return YOLO(model_path)


def get_image_paths(path: Path) -> List[Path]:
    """Get all image files from a path (single file or directory)"""
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        if path.suffix.lower() in supported_exts:
            return [path]
        else:
            raise ValueError(f"File {path} is not a supported image format")
    elif path.is_dir():
        images = []
        for ext in supported_exts:
            images.extend(path.glob(f"**/*{ext}"))
            images.extend(path.glob(f"**/*{ext.upper()}"))
        return sorted(images)
    else:
        raise ValueError(f"Path {path} does not exist")


def run_segmentation_inference(
    image_path: Path,
    model: YOLO,
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Run YOLO segmentation inference on a single image.
    
    Args:
        image_path: Path to the input image
        model: Loaded YOLO segmentation model
        imgsz: Inference image size (default: 640)
        conf: Confidence threshold (default: 0.25)
        
    Returns:
        Tuple of (original_image, segmentation_masks, class_names)
        - original_image: Original input image as numpy array
        - segmentation_masks: Segmentation masks [H, W] or None if no detections
        - class_names: List of detected class names
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
        return orig_img, None, []
    
    result = results[0]
    
    # Get segmentation masks
    masks = result.masks.data.cpu().numpy()  # [N, H, W]
    
    # Combine all masks into a single mask
    if masks.shape[0] > 0:
        # Resize masks to original image size
        combined_mask = np.zeros((orig_img.shape[0], orig_img.shape[1]))
        for i, mask in enumerate(masks):
            # Resize mask to match original image dimensions
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((orig_img.shape[1], orig_img.shape[0]), PILImage.Resampling.LANCZOS)
            mask_resized = np.array(mask_resized) / 255.0
            
            # Add to combined mask with different values for different instances
            combined_mask[mask_resized > 0.5] = i + 1
    else:
        combined_mask = None
    
    # Get class names for detected objects
    detected_classes = []
    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = result.names if hasattr(result, "names") and isinstance(result.names, dict) else model.names
        
        if isinstance(class_names, dict):
            detected_classes = [class_names.get(cls_id, f"class_{cls_id}") for cls_id in class_ids]
        else:
            detected_classes = [class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}" for cls_id in class_ids]
    
    return orig_img, combined_mask, detected_classes


def visualize_segmentation_results(
    original_image: np.ndarray,
    segmentation_mask: Optional[np.ndarray],
    class_names: List[str],
    image_name: str,
) -> None:
    """
    Visualize original image and segmentation results using matplotlib.
    
    Args:
        original_image: Original input image
        segmentation_mask: Segmentation mask or None if no detections
        class_names: List of detected class names
        image_name: Name of the image file for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    else:
        axes[1].text(0.5, 0.5, "No objects detected", ha='center', va='center', fontsize=16, 
                    transform=axes[1].transAxes)
        axes[1].set_title("Segmentation Mask\nNo detections")
        axes[1].axis("off")
        
        axes[2].imshow(original_image)
        axes[2].set_title("Original + Segmentation Overlay\nNo detections")
        axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def run_segmentation_on_images(
    input_path: str,
    model_path: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.25,
) -> None:
    """
    Main function to run segmentation on images from a folder or single image.
    
    Args:
        input_path: Path to input image or folder containing images
        model_path: Path to YOLO segmentation model (default: yolov8s-seg.pt)
        imgsz: Inference image size (default: 640)
        conf: Confidence threshold (default: 0.25)
    """
    print(f"Loading YOLO segmentation model...")
    model = load_segmentation_model(model_path)
    
    # Get list of images to process
    path = Path(input_path)
    image_paths = get_image_paths(path)
    
    if not image_paths:
        raise ValueError(f"No images found at {input_path}")
    
    print(f"Found {len(image_paths)} image(s) to process")
    
    total_start_time = time.time()
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path.name}")
        
        try:
            start_time = time.time()
            
            # Run segmentation inference
            original_img, seg_mask, class_names = run_segmentation_inference(
                image_path=image_path,
                model=model,
                imgsz=imgsz,
                conf=conf,
            )
            
            # Visualize results
            visualize_segmentation_results(
                original_image=original_img,
                segmentation_mask=seg_mask,
                class_names=class_names,
                image_name=image_path.name,
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
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on images and visualize results")
    parser.add_argument("input", type=str, help="Path to input image or folder containing images")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to YOLO segmentation model (default: yolov8s-seg.pt)")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, 
                       help="Confidence threshold (default: 0.25)")
    
    args = parser.parse_args()
    
    run_segmentation_on_images(
        input_path=args.input,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()

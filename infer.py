import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


def parse_kp_names(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    names = [s.strip() for s in arg.split(",")]
    return [n for n in names if n]


def pad_to_square(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad image to square while keeping center, return padded image and padding offsets"""
    h, w = img.shape[:2]
    
    if h == w:
        return img, 0, 0
    
    # Determine the size of the square (largest dimension)
    size = max(h, w)
    
    # Calculate padding needed
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    
    # Handle odd padding
    pad_top = pad_h
    pad_bottom = size - h - pad_top
    pad_left = pad_w  
    pad_right = size - w - pad_left
    
    # Pad the image with white
    if len(img.shape) == 3:
        # For RGB images, pad with white (1.0 for float images, 255 for uint8)
        pad_value = 1.0 if img.max() <= 1.0 else 255
        padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=pad_value)
    else:
        # For grayscale images
        pad_value = 1.0 if img.max() <= 1.0 else 255
        padded = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=pad_value)
    
    return padded, pad_left, pad_top





def draw_prediction(
    ax,
    img_h: int,
    img_w: int,
    boxes_xyxy: np.ndarray,
    classes: Optional[np.ndarray],
    keypoints_xy: Optional[np.ndarray],
    keypoints_conf: Optional[np.ndarray],
    class_names: List[str],
    keypoint_names: Optional[List[str]] = None,
) -> None:
    # Lazy import to avoid matplotlib dependency in non-visual mode
    import matplotlib.patches as patches  # type: ignore
    num_instances = 0 if boxes_xyxy is None else boxes_xyxy.shape[0]
    for i in range(num_instances):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        cls_id = int(classes[i]) if classes is not None else -1
        cls_text = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        ax.text(
            max(0, x1),
            max(0, y1 - 5),
            cls_text,
            color="yellow",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )

        if keypoints_xy is not None:
            kps = keypoints_xy[i]  # [K,2]
            kps_conf = keypoints_conf[i] if keypoints_conf is not None else None  # [K]
            K = kps.shape[0]
            for k in range(K):
                kx, ky = kps[k].tolist()
                v = kps_conf[k] if kps_conf is not None else 1.0
                if v is None:
                    v = 1.0
                if v > 0:  # visible / confident
                    ax.scatter([kx], [ky], c="red", s=30, zorder=3)
                    label_text = str(k)
                    if keypoint_names is not None and k < len(keypoint_names):
                        label_text = keypoint_names[k]
                    ax.text(kx + 2, ky + 2, label_text, color="white", fontsize=8, zorder=4)

            # Connect consecutive visible keypoints (0-1-2-...)
            for k in range(K - 1):
                x1, y1 = kps[k].tolist()
                x2, y2 = kps[k + 1].tolist()
                v1 = kps_conf[k] if kps_conf is not None else 1.0
                v2 = kps_conf[k + 1] if kps_conf is not None else 1.0
                if (v1 is None or v2 is None) or (v1 <= 0 or v2 <= 0):
                    continue
                ax.plot([x1, x2], [y1, y2], color="cyan", linewidth=2)





def inference_single_image(
    image: np.ndarray,
    model: Optional[YOLO] = None,
    imgsz: int = 1024,
    conf: float = 0.25,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Run inference on a single image with hardcoded defaults.
    
    Args:
        image: Input image as numpy array (H, W, 3) in RGB format
        model: Loaded YOLO model (default: loads train11/last.pt)
        imgsz: Inference image size (default: 1024)
        conf: Confidence threshold (default: 0.25)
        
    Returns:
        Tuple of (boxes_xyxy, keypoints_xy, keypoints_conf, classes, class_names_list)
        - boxes_xyxy: Bounding boxes in original image coordinates [N, 4]
        - keypoints_xy: Keypoints in original image coordinates [N, K, 2] 
        - keypoints_conf: Keypoint confidences [N, K]
        - classes: Class IDs [N]
        - class_names_list: List of class names
    """
    # Load default model if not provided
    if model is None:
        default_model_path = Path("runs/pose/train11/weights/last.pt")
        model = YOLO(str(default_model_path))
    orig_h, orig_w = image.shape[0], image.shape[1]
    
    # Convert to RGB if needed
    if image.ndim == 2:  # grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # drop alpha
    
    # Pad the image to square for YOLO inference
    padded_img, pad_x, pad_y = pad_to_square(image)
    padded_h, padded_w = padded_img.shape[0], padded_img.shape[1]
    
    # Save padded image temporarily for YOLO inference
    import tempfile
    from PIL import Image as PILImage
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        padded_pil = PILImage.fromarray((padded_img * 255).astype(np.uint8) if padded_img.max() <= 1.0 else padded_img.astype(np.uint8))
        padded_pil.save(tmp_file.name)
        tmp_path = tmp_file.name
    
    try:
        results = model.predict(source=tmp_path, imgsz=imgsz, conf=conf, verbose=False)
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)
    
    if not results:
        return None, None, None, None, []
    
    result = results[0]
    # Get inference dimensions (should be square now)
    inf_h, inf_w = result.orig_img.shape[0], result.orig_img.shape[1]
    
    # Scale factor from inference to padded image
    scale_x = padded_w / inf_w
    scale_y = padded_h / inf_h
    
    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    classes = result.boxes.cls.cpu().numpy().astype(int) if (result.boxes is not None and result.boxes.cls is not None) else None
    
    kp = result.keypoints
    keypoints_xy = kp.xy.cpu().numpy() if kp is not None and kp.xy is not None else None
    keypoints_conf = kp.conf.cpu().numpy() if kp is not None and hasattr(kp, "conf") and kp.conf is not None else None
    
    # Scale boxes to padded image resolution, then adjust for padding to get original coordinates
    if boxes_xyxy is not None:
        boxes_xyxy = boxes_xyxy * [scale_x, scale_y, scale_x, scale_y]
        # Subtract padding to get coordinates in original image space
        boxes_xyxy = boxes_xyxy - [pad_x, pad_y, pad_x, pad_y]
    
    # Scale keypoints to padded image resolution, then adjust for padding
    if keypoints_xy is not None:
        keypoints_xy = keypoints_xy * [scale_x, scale_y]
        # Subtract padding to get coordinates in original image space
        keypoints_xy = keypoints_xy - [pad_x, pad_y]
    
    # Get class names
    class_names = result.names if hasattr(result, "names") and isinstance(result.names, dict) else model.names
    if isinstance(class_names, dict):
        # Convert dict {id: name} to list indexed by id
        max_id = max(class_names.keys()) if class_names else -1
        names_list = [""] * (max_id + 1)
        for k, v in class_names.items():
            names_list[int(k)] = str(v)
        class_names_list = names_list
    else:
        class_names_list = list(class_names)
    
    return boxes_xyxy, keypoints_xy, keypoints_conf, classes, class_names_list


def run_inference(
    image_path: Path,
    model_path: Path,
    imgsz: int,
    conf: float,
    keypoint_names: Optional[List[str]] = None,
) -> float:
    start_time = time.time()

    # Load original high-res image for plotting
    import matplotlib.image as mpimg
    orig_img = mpimg.imread(str(image_path))
    if orig_img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]

    # Load model and run inference using the single method
    model = YOLO(str(model_path))
    boxes_xyxy, keypoints_xy, keypoints_conf, classes, class_names_list = inference_single_image(
        orig_img, model, imgsz, conf
    )
    
    # Get padded image for visualization
    padded_img, pad_x, pad_y = pad_to_square(orig_img)

    # Always show visualization
    import matplotlib.pyplot as plt

    # Scale boxes and keypoints for the manually created 1024x1024 image
    yolo_scaled_boxes = None
    yolo_scaled_keypoints = None
    if boxes_xyxy is not None:
        # Add padding back to get padded coordinates, then scale to imgsz
        temp_boxes = boxes_xyxy + [pad_x, pad_y, pad_x, pad_y]
        # Scale from padded image size to imgsz (e.g., 1024x1024)
        padded_size = padded_img.shape[0]  # Should be square
        scale_to_imgsz = imgsz / padded_size
        yolo_scaled_boxes = temp_boxes * scale_to_imgsz
    if keypoints_xy is not None:
        # Add padding back to get padded coordinates, then scale to imgsz
        temp_keypoints = keypoints_xy + [pad_x, pad_y]
        # Scale from padded image size to imgsz (e.g., 1024x1024)
        padded_size = padded_img.shape[0]  # Should be square
        scale_to_imgsz = imgsz / padded_size
        yolo_scaled_keypoints = temp_keypoints * scale_to_imgsz


    
    # Create the actual scaled-down image that YOLO processes (e.g., 1024x1024)
    from PIL import Image as PILImage
    padded_pil = PILImage.fromarray((padded_img * 255).astype(np.uint8) if padded_img.max() <= 1.0 else padded_img.astype(np.uint8))
    yolo_scaled_pil = padded_pil.resize((imgsz, imgsz), PILImage.Resampling.LANCZOS)
    yolo_scaled_img = np.array(yolo_scaled_pil)

    # Create figure with four subplots: original, padded full-res, YOLO scaled, original with labels
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    if boxes_xyxy is not None and keypoints_xy is not None and boxes_xyxy.shape[0] > 0:
        # Top-left: original image without labels
        ax1.imshow(orig_img)
        ax1.set_title(f"Original Image ({orig_w}x{orig_h})")
        ax1.axis("off")

        # Top-right: padded square full resolution
        ax2.imshow(padded_img)
        ax2.set_title(f"Padded Square - Full Resolution ({padded_img.shape[1]}x{padded_img.shape[0]})")
        ax2.axis("off")

        # Bottom-left: actual YOLO input (scaled down, e.g., 1024x1024) with predictions
        ax3.imshow(yolo_scaled_img)
        ax3.set_title(f"YOLO Input - Scaled ({imgsz}x{imgsz}) with Labels")
        ax3.axis("off")
        draw_prediction(
            ax=ax3,
            img_h=imgsz,
            img_w=imgsz,
            boxes_xyxy=yolo_scaled_boxes,
            classes=classes,
            keypoints_xy=yolo_scaled_keypoints,
            keypoints_conf=keypoints_conf,
            class_names=class_names_list,
            keypoint_names=keypoint_names,
        )

        # Bottom-right: original high-res with predictions
        ax4.imshow(orig_img)
        ax4.set_title(f"Original with Labels ({orig_w}x{orig_h})")
        ax4.axis("off")
        draw_prediction(
            ax=ax4,
            img_h=orig_h,
            img_w=orig_w,
            boxes_xyxy=boxes_xyxy,
            classes=classes,
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            class_names=class_names_list,
            keypoint_names=keypoint_names,
        )

        plt.tight_layout()
        plt.show()
        plt.close(fig)
    else:
            # Handle case with no detections
        # Top-left: original image
        ax1.imshow(orig_img)
        ax1.set_title(f"Original Image ({orig_w}x{orig_h}) - No detections")
        ax1.axis("off")
        
        # Top-right: padded square full resolution
        ax2.imshow(padded_img)
        ax2.set_title(f"Padded Square - Full Resolution ({padded_img.shape[1]}x{padded_img.shape[0]}) - No detections")
        ax2.axis("off")
        
        # Bottom-left: YOLO input scaled
        ax3.imshow(yolo_scaled_img)
        ax3.set_title(f"YOLO Input - Scaled ({imgsz}x{imgsz}) - No detections")
        ax3.axis("off")
        
        # Bottom-right: original (same as top-left since no labels)
        ax4.imshow(orig_img)
        ax4.set_title(f"Original with Labels ({orig_w}x{orig_h}) - No detections")
        ax4.axis("off")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    end_time = time.time()
    processing_time = end_time - start_time
    return processing_time


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLOv8-pose inference on images and visualize. Processes images one by one - close each plot window to continue to the next.")
    p.add_argument("--image", type=str, help="Path to input image (mutually exclusive with --folder)")
    p.add_argument("--folder", type=str, help="Path to folder containing images (mutually exclusive with --image)")
    p.add_argument(
        "--model",
        type=str,
        default=str(Path("runs") / "pose" / "train11" / "weights" / "last.pt"),
        help="Path to trained YOLOv8-pose model (default: runs/pose/train11/weights/last.pt)",
    )
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size (default: 1024)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument(
        "--kp-names",
        type=str,
        default="left-out,left-eye,left-edge,right-edge,right-eye,right-out",
        help="Comma-separated keypoint names in order (default: left-out,left-eye,left-edge,right-edge,right-eye,right-out)",
    )
    return p


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


def main() -> None:
    args = build_arg_parser().parse_args()

    # Validate arguments
    if not args.image and not args.folder:
        raise ValueError("Must provide either --image or --folder")
    if args.image and args.folder:
        raise ValueError("Cannot provide both --image and --folder")

    # Determine input path
    input_path = Path(args.image or args.folder)
    model_path = Path(args.model)
    kp_names = parse_kp_names(args.kp_names)

    # Get list of images to process
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No images found at {input_path}")

    print(f"Processing {len(image_paths)} image(s) in visual mode...")

    total_start_time = time.time()

    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path.name}")

        try:
            processing_time = run_inference(
                image_path=image_path,
                model_path=model_path,
                imgsz=args.imgsz,
                conf=args.conf,
                keypoint_names=kp_names,
            )
            print(f"Processing time: {processing_time:.1f}s")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\nFinished processing {len(image_paths)} image(s) in {total_time:.1f}s")


if __name__ == "__main__":
    main()



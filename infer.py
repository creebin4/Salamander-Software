import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from PIL import Image


def parse_kp_names(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    names = [s.strip() for s in arg.split(",")]
    return [n for n in names if n]


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


def compute_angle_deg_from_kps(
    kps_xy: np.ndarray,
    image_center_xy: Optional[Tuple[float, float]],
) -> float:
    if kps_xy.shape[0] < 3:
        return 0.0
    center = np.array([image_center_xy[0], image_center_xy[1]])
    left_eye = kps_xy[0]
    nose = kps_xy[1]
    right_eye = kps_xy[2]
    midpoint = (left_eye + right_eye) / 2
    MC = midpoint - center
    MN  = midpoint - nose
    cos_theta = np.dot(MC, MN) / (np.linalg.norm(MC) * np.linalg.norm(MN))
    theta = np.arccos(cos_theta)
    target_angle = np.degrees(theta)
    print(f"Target angle: {target_angle:.1f}°")

    CM = center - midpoint
    current_angle = 180 - np.degrees(np.arctan2(CM[1], CM[0]))
    print(f"Current angle: {current_angle:.1f}°")
    return target_angle - current_angle + 180

def rotate_and_crop(
    img: np.ndarray,
    center_xy: Tuple[float, float],
    width: float,
    height: float,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    # Rotate whole image around center, then crop axis-aligned rectangle
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for rotate_and_crop") from e

    h, w = img.shape[:2]
    cx, cy = center_xy
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Compute crop bounds in rotated image
    half_w = width / 2.0
    half_h = height / 2.0
    x1 = int(round(cx - half_w))
    y1 = int(round(cy - half_h))
    x2 = int(round(cx + half_w))
    y2 = int(round(cy + half_h))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return rotated[0:1, 0:1].copy(), M, x1, y1
    return rotated[y1:y2, x1:x2].copy(), M, x1, y1


def apply_affine_to_point(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return x_new, y_new


def run_inference(
    image_path: Path,
    model_path: Path,
    imgsz: int,
    conf: float,
    keypoint_names: Optional[List[str]] = None,
    visual_mode: bool = True,
    output_path: Optional[Path] = None,
) -> None:
    # Load original high-res image for plotting and cropping
    orig_img = mpimg.imread(str(image_path))
    if orig_img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    orig_h, orig_w = orig_img.shape[0], orig_img.shape[1]

    # Convert to RGB if needed (matplotlib.imread handles most formats)
    if orig_img.ndim == 2:  # grayscale
        orig_img = np.stack([orig_img] * 3, axis=-1)
    elif orig_img.shape[2] == 4:  # RGBA
        orig_img = orig_img[:, :, :3]  # drop alpha

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        raise RuntimeError("No results from model.predict")

    result = results[0]
    # Get inference dimensions
    inf_h, inf_w = result.orig_img.shape[0], result.orig_img.shape[1]

    # Scale factor from inference to original
    scale_x = orig_w / inf_w
    scale_y = orig_h / inf_h

    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    classes = result.boxes.cls.cpu().numpy().astype(int) if (result.boxes is not None and result.boxes.cls is not None) else None

    kp = result.keypoints
    keypoints_xy = kp.xy.cpu().numpy() if kp is not None and kp.xy is not None else None
    keypoints_conf = kp.conf.cpu().numpy() if kp is not None and hasattr(kp, "conf") and kp.conf is not None else None

    # Scale boxes to original resolution
    if boxes_xyxy is not None:
        boxes_xyxy = boxes_xyxy * [scale_x, scale_y, scale_x, scale_y]

    # Scale keypoints to original resolution
    if keypoints_xy is not None:
        keypoints_xy = keypoints_xy * [scale_x, scale_y]

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

    if boxes_xyxy is not None and keypoints_xy is not None and boxes_xyxy.shape[0] > 0:
        # Prepare cropped image from original high-res
        x1, y1, x2, y2 = boxes_xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1)
        bh = (y2 - y1)
        # Use original image center as reference (optional)
        angle = compute_angle_deg_from_kps(keypoints_xy[0], image_center_xy=(cx, cy))
        crop, M, x_off, y_off = rotate_and_crop(orig_img, (cx, cy), bw, bh, angle)

        # Get the downscaled version from model inference
        downscaled_img_bgr = result.orig_img
        downscaled_img = downscaled_img_bgr[:, :, ::-1].copy()  # BGR to RGB

        # Scale boxes and keypoints to downscaled resolution for plotting
        downscaled_boxes = boxes_xyxy / [scale_x, scale_y, scale_x, scale_y] if boxes_xyxy is not None else None
        downscaled_keypoints = keypoints_xy / [scale_x, scale_y] if keypoints_xy is not None else None

        # Create figure with three subplots (original, downscaled, rotated crop)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Left: original high-res with predictions
        ax1.imshow(orig_img)
        ax1.set_title(f"Original ({orig_w}x{orig_h})")
        ax1.axis("off")
        draw_prediction(
            ax=ax1,
            img_h=orig_h,
            img_w=orig_w,
            boxes_xyxy=boxes_xyxy,
            classes=classes,
            keypoints_xy=keypoints_xy,
            keypoints_conf=keypoints_conf,
            class_names=class_names_list,
            keypoint_names=keypoint_names,
        )

        # Compute midpoint 0-2 and draw guide to point 1 on original and downscaled
        if keypoints_xy is not None and keypoints_xy.shape[1] >= 3:
            x0, y0 = keypoints_xy[0][0].tolist()
            x1k, y1k = keypoints_xy[0][1].tolist()
            x2, y2 = keypoints_xy[0][2].tolist()
            mx, my = (x0 + x2) / 2.0, (y0 + y2) / 2.0
            # Original
            ax1.scatter([mx], [my], c="orange", s=40, zorder=5)
            ax1.plot([mx, x1k], [my, y1k], color="orange", linewidth=2, zorder=5)

            # Downscaled
            if downscaled_keypoints is not None:
                x0d, y0d = downscaled_keypoints[0][0].tolist()
                x1d, y1d = downscaled_keypoints[0][1].tolist()
                x2d, y2d = downscaled_keypoints[0][2].tolist()
                mxd, myd = (x0d + x2d) / 2.0, (y0d + y2d) / 2.0
                ax2.scatter([mxd], [myd], c="orange", s=40, zorder=5)
                ax2.plot([mxd, x1d], [myd, y1d], color="orange", linewidth=2, zorder=5)

        # Middle: downscaled version with predictions
        ax2.imshow(downscaled_img)
        ax2.set_title(f"Model Input ({inf_w}x{inf_h})")
        ax2.axis("off")
        draw_prediction(
            ax=ax2,
            img_h=inf_h,
            img_w=inf_w,
            boxes_xyxy=downscaled_boxes,
            classes=classes,
            keypoints_xy=downscaled_keypoints,
            keypoints_conf=keypoints_conf,
            class_names=class_names_list,
            keypoint_names=keypoint_names,
        )

        # Right: cropped rotated high-res image
        ax3.imshow(crop)
        ax3.set_title(f"Rotated Crop ({crop.shape[1]}x{crop.shape[0]}) | Vector 0->2 Down | Angle: {angle:.1f}°")
        ax3.axis("off")

        # Draw midpoint and line on rotated crop (transform by M and offset by crop top-left)
        if keypoints_xy is not None and keypoints_xy.shape[1] >= 3:
            x0, y0 = keypoints_xy[0][0].tolist()
            x1k, y1k = keypoints_xy[0][1].tolist()
            x2, y2 = keypoints_xy[0][2].tolist()
            mx, my = (x0 + x2) / 2.0, (y0 + y2) / 2.0
            rx_m, ry_m = apply_affine_to_point(M, mx, my)
            rx_1, ry_1 = apply_affine_to_point(M, x1k, y1k)
            # shift into crop coordinates
            rx_m -= x_off
            ry_m -= y_off
            rx_1 -= x_off
            ry_1 -= y_off
            ax3.scatter([rx_m], [ry_m], c="orange", s=40, zorder=5)
            ax3.plot([rx_m, rx_1], [ry_m, ry_1], color="orange", linewidth=2, zorder=5)

        if visual_mode:
            plt.tight_layout()
            plt.show()
        else:
            # Save the processed crop image
            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / image_path.name
                # Convert numpy array to PIL Image and save
                crop_pil = Image.fromarray(crop)
                crop_pil.save(str(output_file))
                print(f"Saved processed image to: {output_file}")
    else:
        if visual_mode:
            # No detections, show original and downscaled
            downscaled_img_bgr = result.orig_img
            downscaled_img = downscaled_img_bgr[:, :, ::-1].copy()  # BGR to RGB

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.imshow(orig_img)
            ax1.set_title(f"Original ({orig_w}x{orig_h}) | No detections")
            ax1.axis("off")
            ax2.imshow(downscaled_img)
            ax2.set_title(f"Model Input ({inf_w}x{inf_h}) | No detections")
            ax2.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"No detections found for {image_path.name}, skipping save")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLOv8-pose inference on images and visualize. Processes images one by one - close each plot window to continue to the next.")
    p.add_argument("--image", type=str, help="Path to input image (mutually exclusive with --folder)")
    p.add_argument("--folder", type=str, help="Path to folder containing images (mutually exclusive with --image)")
    p.add_argument(
        "--model",
        type=str,
        default=str(Path("runs") / "pose" / "train5" / "weights" / "best.pt"),
        help="Path to trained YOLOv8-pose model",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument(
        "--kp-names",
        type=str,
        default="left-eye,snout,right-eye",
        help="Comma-separated keypoint names in order. Leave empty to show indices.",
    )
    p.add_argument("--visual", action="store_true", help="Show visualization instead of saving processed images")
    p.add_argument("--output", type=str, help="Output folder path for processed images (required when not using --visual)")
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

    # Validate visual/output arguments
    if not args.visual and not args.output:
        raise ValueError("Must specify --output when not using --visual mode")
    if args.visual and args.output:
        raise ValueError("Cannot specify --output when using --visual mode")

    # Determine input path
    input_path = Path(args.image or args.folder)
    model_path = Path(args.model)
    kp_names = parse_kp_names(args.kp_names)
    output_path = Path(args.output) if args.output else None

    # Get list of images to process
    image_paths = get_image_paths(input_path)
    if not image_paths:
        raise ValueError(f"No images found at {input_path}")

    print(f"Processing {len(image_paths)} image(s)...")

    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path.name}")
        try:
            run_inference(
                image_path=image_path,
                model_path=model_path,
                imgsz=args.imgsz,
                conf=args.conf,
                keypoint_names=kp_names,
                visual_mode=args.visual,
                output_path=output_path,
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print(f"\nFinished processing {len(image_paths)} image(s)")


if __name__ == "__main__":
    main()



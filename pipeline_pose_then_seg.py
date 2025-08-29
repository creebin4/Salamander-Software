import argparse
from pathlib import Path
from typing import List, Optional

from rotate_and_crop import process_and_crop_image
from segmentation_crop import (
    load_segmentation_models,
    run_segmentation_inference,
    select_largest_instance_mask,
    largest_instance_in_center,
    compute_top_crop_using_middle_vertical_third,
    compute_bottom_crop_using_middle_vertical_third,
    compute_right_crop_using_middle_horizontal_third,
    crop_image_using_margins,
)
from yolo_utils import list_image_files, save_image


def process_single_image(
    image_path: Path,
    output_dir: Path,
    pose_model_path: Optional[str],
    seg_models: List,
    pose_imgsz: int = 1024,
    pose_conf: float = 0.25,
    seg_imgsz: int = 640,
    seg_conf: float = 0.25,
    tb_add_back_pct: float = 5.0,
) -> Optional[Path]:
    # Write pose-aligned crop directly to output path with original filename (preserve name and extension)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / image_path.name

    aligned_crop = process_and_crop_image(
        image_path=image_path,
        visual=False,
        output_path=out_path,
        model_path=pose_model_path,
    )
    if aligned_crop is None:
        print("  ✗ No pose detections; skipping")
        return None

    # Now run segmentation on the saved aligned crop and overwrite the same file
    original_img = None
    instance_masks_all = []
    for model in seg_models:
        img, combined_mask, _, instance_masks_resized, _ = run_segmentation_inference(
            image_path=out_path,
            model=model,
            imgsz=seg_imgsz,
            conf=seg_conf,
        )
        if original_img is None:
            original_img = img
        if combined_mask is not None and len(instance_masks_resized) > 0 and largest_instance_in_center(
            instance_masks_resized, img.shape[:2]
        ):
            instance_masks_all = instance_masks_resized
            break

    largest_mask = select_largest_instance_mask(instance_masks_all)
    if largest_mask is not None:
        top_crop = compute_top_crop_using_middle_vertical_third(largest_mask)
        bottom_crop = compute_bottom_crop_using_middle_vertical_third(largest_mask)
        right_crop = compute_right_crop_using_middle_horizontal_third(largest_mask)
        h, _ = original_img.shape[:2]
        if tb_add_back_pct and tb_add_back_pct > 0:
            add_back_px = int(round((tb_add_back_pct / 100.0) * h))
            top_crop = max(0, top_crop - add_back_px)
            bottom_crop = max(0, bottom_crop - add_back_px)
        final_img = crop_image_using_margins(original_img, top_crop, bottom_crop, right_crop)
        save_image(out_path, final_img)
    # If no mask found, the pose-aligned image already exists at out_path
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose rotate+crop then segmentation precise crop using existing methods")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--folder", type=str, help="Path to folder of images")

    parser.add_argument("--pose-model", type=str, default="last.pt", help="Path to YOLOv8 pose model")
    parser.add_argument("--pose-imgsz", type=int, default=1024, help="Pose inference image size (kept for compatibility)")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="Pose confidence threshold (kept for compatibility)")

    parser.add_argument("--seg-model", type=str, default=None, help="Primary YOLO segmentation model (default yolov8s-seg.pt)")
    parser.add_argument("--seg-imgsz", type=int, default=640, help="Segmentation inference image size")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="Segmentation confidence threshold")
    parser.add_argument("--tb-add-back-pct", type=float, default=5.0, help="Add back percentage of image height to top/bottom crops")

    parser.add_argument("--output", type=str, default="./output", help="Output directory for final crops")

    args = parser.parse_args()

    # Build segmentation model search order (same as segmentation_crop)
    primary = args.seg_model if args.seg_model is not None else "yolov8s-seg.pt"
    fallbacks: List[str] = [
        "yolov8x-seg.pt",
        "yolov9c-seg.pt",
        "yolov9e-seg.pt",
        "yolo11n-seg.pt",
        "yolo11s-seg.pt",
        "yolo11x-seg.pt",
    ]
    seg_models_paths: List[str] = []
    seen = set()
    for m in [primary] + fallbacks:
        if m not in seen:
            seg_models_paths.append(m)
            seen.add(m)
    seg_models = load_segmentation_models(seg_models_paths)
    if not seg_models:
        raise RuntimeError("No valid segmentation models could be loaded.")

    input_path = Path(args.image) if args.image else Path(args.folder)
    images = list_image_files(input_path) if input_path.is_dir() else [input_path]
    images = [p for p in images if p.exists()]
    if not images:
        raise RuntimeError(f"No supported images found at {input_path}")

    output_dir = Path(args.output)
    print(f"Found {len(images)} image(s) to process")
    success_count = 0
    fail_count = 0

    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name}")
        try:
            out_path = process_single_image(
                image_path=img_path,
                output_dir=output_dir,
                pose_model_path=args.pose_model,
                seg_models=seg_models,
                pose_imgsz=args.pose_imgsz,
                pose_conf=args.pose_conf,
                seg_imgsz=args.seg_imgsz,
                seg_conf=args.seg_conf,
                tb_add_back_pct=args.tb_add_back_pct,
            )
            if out_path is not None:
                success_count += 1
                print(f"  ✓ Saved: {out_path}")
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Error: {e}")

    print(f"\nDone. Successful: {success_count}, Failed: {fail_count}. Output: {output_dir}")


if __name__ == "__main__":
    main()



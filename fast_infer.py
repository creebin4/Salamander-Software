import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO
from PIL import Image


def get_image_paths(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def rotate_and_crop(img: np.ndarray, center_xy: Tuple[float, float], width: float, height: float, angle_deg: float) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for rotate_and_crop") from e
    h, w = img.shape[:2]
    cx, cy = center_xy

    # Stage 1: pre-crop an axis-aligned region 2x the requested width/height around the center
    pre_w = max(1, int(round(2.0 * width)))
    pre_h = max(1, int(round(2.0 * height)))
    x1 = max(0, int(round(cx - pre_w / 2.0)))
    y1 = max(0, int(round(cy - pre_h / 2.0)))
    x2 = min(w, int(round(cx + pre_w / 2.0)))
    y2 = min(h, int(round(cy + pre_h / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1].copy()

    sub = img[y1:y2, x1:x2]

    # Recenter coordinates relative to the pre-crop
    sub_cx = cx - x1
    sub_cy = cy - y1

    # Stage 2: rotate the pre-cropped region and warp directly to final (width x height)
    out_w = max(1, int(round(width)))
    out_h = max(1, int(round(height)))
    M = cv2.getRotationMatrix2D((sub_cx, sub_cy), angle_deg, 1.0)
    M[0, 2] += (out_w / 2.0 - sub_cx)
    M[1, 2] += (out_h / 2.0 - sub_cy)
    out = cv2.warpAffine(
        sub,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return out


def compute_angle_deg_from_kps(kps_xy: np.ndarray, image_center_xy: Tuple[float, float]) -> float:
    if kps_xy.shape[0] < 3:
        return 0.0
    center = np.array([image_center_xy[0], image_center_xy[1]])
    left_eye = kps_xy[0]
    nose = kps_xy[1]
    right_eye = kps_xy[2]
    midpoint = (left_eye + right_eye) / 2
    MC = midpoint - center
    MN = midpoint - nose
    cos_theta = np.dot(MC, MN) / (np.linalg.norm(MC) * np.linalg.norm(MN))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    target_angle = np.degrees(theta)
    CM = center - midpoint
    current_angle = 180 - np.degrees(np.arctan2(CM[1], CM[0]))
    return target_angle - current_angle + 180


def main() -> None:
    input_root = Path("./images")
    output_root = Path("./output")
    model_path = Path("runs/pose/train5/weights/best.pt")
    imgsz = 1024
    conf = 0.25
    batch_size = 8
    use_fp16 = True

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    model_load_t0 = time.time()
    model = YOLO(str(model_path))
    model_load_t1 = time.time()

    image_paths = get_image_paths(input_root)
    if not image_paths:
        raise RuntimeError(f"No images found under {input_root}")

    total_predict_s = 0.0
    total_post_s = 0.0
    total_io_write_s = 0.0
    per_image_times = []

    print(f"Found {len(image_paths)} image(s). Model load: {(model_load_t1 - model_load_t0):.3f}s")

    # Process in batches to leverage vectorized inference
    for b_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[b_start : b_start + batch_size]
        batch_strs = [str(p) for p in batch_paths]

        t_pred0 = time.time()
        results = model.predict(
            source=batch_strs,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            batch=batch_size,
            half=use_fp16,
        )
        t_pred1 = time.time()
        batch_pred_s = (t_pred1 - t_pred0)
        total_predict_s += batch_pred_s

        # Post-process each result
        for i, r in enumerate(results):
            path = batch_paths[i]
            t_post0 = time.time()
            orig_bgr = r.orig_img  # BGR
            img_np = orig_bgr[:, :, ::-1].copy()  # to RGB

            boxes_xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            kps = r.keypoints
            keypoints_xy = kps.xy.cpu().numpy() if kps is not None and kps.xy is not None else None

            if boxes_xyxy is not None and keypoints_xy is not None and boxes_xyxy.shape[0] > 0:
                x1, y1, x2, y2 = boxes_xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1)
                bh = (y2 - y1)
                angle = compute_angle_deg_from_kps(keypoints_xy[0], image_center_xy=(cx, cy))
                crop = rotate_and_crop(img_np, (cx, cy), bw, bh, angle)
            else:
                crop = img_np[0:1, 0:1].copy()
            t_post1 = time.time()
            total_post_s += (t_post1 - t_post0)

            t_io0 = time.time()
            out_path = output_root / path.name
            Image.fromarray(crop).save(str(out_path))
            t_io1 = time.time()
            total_io_write_s += (t_io1 - t_io0)

            approx_pred_per_img = batch_pred_s / max(1, len(results))
            per_image_times.append((path.name, 0.0, approx_pred_per_img, t_post1 - t_post0, t_io1 - t_io0))
            idx_global = b_start + i + 1
            print(f"[{idx_global}/{len(image_paths)}] {path.name}: predict~={approx_pred_per_img:.3f}s, post={(t_post1 - t_post0):.3f}s, write={(t_io1 - t_io0):.3f}s")

    total_images = len(image_paths)
    total_time = (model_load_t1 - model_load_t0) + total_predict_s + total_post_s + total_io_write_s
    print("\n=== Timing Summary ===")
    print(f"Images: {total_images}")
    print(f"Model load: {(model_load_t1 - model_load_t0):.3f}s")
    print(f"Predict: {total_predict_s:.3f}s ({total_predict_s/total_images:.3f}s/img)")
    print(f"Postproc: {total_post_s:.3f}s ({total_post_s/total_images:.3f}s/img)")
    print(f"I/O write: {total_io_write_s:.3f}s ({total_io_write_s/total_images:.3f}s/img)")
    print(f"Total: {total_time:.3f}s ({total_time/total_images:.3f}s/img)")


if __name__ == "__main__":
    main()



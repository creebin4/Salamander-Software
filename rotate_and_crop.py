#!/usr/bin/env python3
"""
Rotate and crop script: Load image, run inference, show results and cropped bounding box
"""

import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression
import cv2
from PIL import Image as PILImage
import os

# Import the inference function from infer.py
from infer import inference_single_image, draw_prediction


def crop_bounding_box(image: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """
    Crop image around bounding box with no padding
    
    Args:
        image: Input image as numpy array (H, W, 3)
        box_xyxy: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    
    # Ensure bounds are within image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop the image
    cropped = image[y1:y2, x1:x2]
    return cropped


def compute_line_of_best_fit(keypoints: np.ndarray, keypoints_conf: np.ndarray = None, conf_threshold: float = 0.5):
    """
    Compute line of best fit through midpoints of keypoint pairs (5&0, 4&1, 3&2)
    
    Args:
        keypoints: Keypoints array [K, 2] for single detection (expects 6 keypoints)
        keypoints_conf: Keypoint confidences [K] (optional)
        conf_threshold: Minimum confidence to include keypoint pair
        
    Returns:
        Tuple of (slope, intercept, x_range, y_range) or None if not enough points
    """
    if keypoints is None or len(keypoints) < 6:
        return None
    
    # Define keypoint pairs: (5&0, 4&1, 3&2)
    keypoint_pairs = [(5, 0), (4, 1), (3, 2)]
    midpoints = []
    
    for idx1, idx2 in keypoint_pairs:
        # Check if both keypoints in the pair are confident enough
        if keypoints_conf is not None:
            conf1 = keypoints_conf[idx1] if idx1 < len(keypoints_conf) else 0.0
            conf2 = keypoints_conf[idx2] if idx2 < len(keypoints_conf) else 0.0
            if conf1 < conf_threshold or conf2 < conf_threshold:
                continue  # Skip this pair if either point has low confidence
        
        # Calculate midpoint between the two keypoints
        kp1 = keypoints[idx1]
        kp2 = keypoints[idx2]
        midpoint = (kp1 + kp2) / 2.0
        midpoints.append(midpoint)
    
    if len(midpoints) < 2:
        return None  # Need at least 2 midpoints for a line
    
    midpoints = np.array(midpoints)
    
    # Extract x and y coordinates of midpoints
    x_coords = midpoints[:, 0].reshape(-1, 1)
    y_coords = midpoints[:, 1]
    
    # Fit linear regression through midpoints
    reg = LinearRegression()
    reg.fit(x_coords, y_coords)
    
    # Get slope and intercept
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # Calculate x range for line plotting (extend beyond midpoints)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_range = np.array([x_min - 50, x_max + 50])
    y_range = slope * x_range + intercept
    
    return slope, intercept, x_range, y_range


def draw_line_of_best_fit(ax, keypoints: np.ndarray, keypoints_conf: np.ndarray = None, show_midpoints: bool = True):
    """
    Draw line of best fit through midpoints on the given axes
    
    Args:
        ax: Matplotlib axes object
        keypoints: Keypoints array [K, 2] for single detection
        keypoints_conf: Keypoint confidences [K] (optional)
        show_midpoints: Whether to show the midpoints used for regression
    """
    if keypoints is None or len(keypoints) < 6:
        return
    
    # Calculate midpoints for visualization
    keypoint_pairs = [(5, 0), (4, 1), (3, 2)]
    midpoints = []
    conf_threshold = 0.5
    
    for idx1, idx2 in keypoint_pairs:
        # Check confidence
        if keypoints_conf is not None:
            conf1 = keypoints_conf[idx1] if idx1 < len(keypoints_conf) else 0.0
            conf2 = keypoints_conf[idx2] if idx2 < len(keypoints_conf) else 0.0
            if conf1 < conf_threshold or conf2 < conf_threshold:
                continue
        
        # Calculate midpoint
        kp1 = keypoints[idx1]
        kp2 = keypoints[idx2]
        midpoint = (kp1 + kp2) / 2.0
        midpoints.append(midpoint)
    
    if len(midpoints) < 2:
        return
    
    # Draw the midpoints
    if show_midpoints:
        midpoints_array = np.array(midpoints)
        ax.scatter(midpoints_array[:, 0], midpoints_array[:, 1], 
                  c='orange', s=60, marker='x', linewidth=3, 
                  zorder=5, label='Midpoints')
    
    # Draw the line of best fit
    line_data = compute_line_of_best_fit(keypoints, keypoints_conf)
    if line_data is not None:
        slope, intercept, x_range, y_range = line_data
        ax.plot(x_range, y_range, color='magenta', linewidth=3, linestyle='-', alpha=0.8, label='Line of Best Fit')
        
        # Add text showing slope
        mid_x = np.mean(x_range)
        mid_y = slope * mid_x + intercept
        ax.text(mid_x, mid_y - 20, f'Slope: {slope:.3f}', 
                color='magenta', fontsize=10, ha='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def determine_head_direction(keypoints: np.ndarray, keypoints_conf: np.ndarray = None, slope: float = 0.0, intercept: float = 0.0) -> float:
    """
    Determine which direction along the regression line represents the head
    
    Keypoint order: left-out, left-eye, left-edge, right-edge, right-eye, right-out
    - Back of skull: keypoints 0, 5 (left-out, right-out)
    - Middle (eyes): keypoints 1, 4 (left-eye, right-eye)  
    - Front of head: keypoints 2, 3 (left-edge, right-edge)
    
    Args:
        keypoints: Keypoints array [K, 2] for single detection
        keypoints_conf: Keypoint confidences [K] (optional)
        slope: Slope of the regression line
        intercept: Intercept of the regression line
        
    Returns:
        Direction multiplier: 1.0 for normal direction, -1.0 for flipped
    """
    if keypoints is None or len(keypoints) < 6:
        return 1.0
    
    conf_threshold = 0.3
    
    # Front of head keypoints are 2, 3 (left-edge, right-edge)
    front_indices = [2, 3]  # left-edge, right-edge
    valid_front_points = []
    for i in front_indices:
        if keypoints_conf is None or keypoints_conf[i] >= conf_threshold:
            valid_front_points.append(keypoints[i])
    
    # Back of skull keypoints are 0, 5 (left-out, right-out)
    back_indices = [0, 5]  # left-out, right-out
    valid_back_points = []
    for i in back_indices:
        if keypoints_conf is None or keypoints_conf[i] >= conf_threshold:
            valid_back_points.append(keypoints[i])
    
    # If we don't have both front and back points, use eyes as backup
    if len(valid_front_points) == 0 or len(valid_back_points) == 0:
        # Use eyes (middle) vs back as fallback
        eye_indices = [1, 4]  # left-eye, right-eye
        valid_eye_points = []
        for i in eye_indices:
            if keypoints_conf is None or keypoints_conf[i] >= conf_threshold:
                valid_eye_points.append(keypoints[i])
        
        if len(valid_eye_points) == 0 or len(valid_back_points) == 0:
            return 1.0  # Default direction if insufficient keypoints
        
        # Use eyes as "front" reference
        front_center = np.mean(valid_eye_points, axis=0)
    else:
        # Use actual front points
        front_center = np.mean(valid_front_points, axis=0)
    
    # Calculate average back position
    back_center = np.mean(valid_back_points, axis=0)
    
    # Calculate the actual back-to-front vector (proper forward direction)
    actual_direction = front_center - back_center
    
    # Calculate the regression line direction vector
    # For a line y = slope * x + intercept, direction vector is (1, slope)
    line_direction = np.array([1.0, slope])
    
    # Normalize both vectors
    actual_direction = actual_direction / (np.linalg.norm(actual_direction) + 1e-8)
    line_direction = line_direction / (np.linalg.norm(line_direction) + 1e-8)
    
    # Check if vectors are pointing in similar direction (dot product > 0)
    dot_product = np.dot(actual_direction, line_direction)
    
    # If dot product is negative, the line direction is opposite to back-to-front direction
    return 1.0 if dot_product >= 0 else -1.0


def rotate_and_crop_aligned(
    image: np.ndarray, 
    bounding_box: np.ndarray, 
    keypoints: np.ndarray, 
    keypoints_conf: np.ndarray = None
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """
    Rotate image to align body axis horizontally and crop to square
    
    Args:
        image: Input image as numpy array (H, W, 3)
        bounding_box: Bounding box as [x1, y1, x2, y2]
        keypoints: Keypoints array [K, 2] for single detection
        keypoints_conf: Keypoint confidences [K] (optional)
        
    Returns:
        Tuple of (rotated_cropped_image, rotation_angle_degrees, rotation_center)
    """
    # Get line of best fit through midpoints
    line_data = compute_line_of_best_fit(keypoints, keypoints_conf)
    if line_data is None:
        # Fallback: just crop without rotation
        return crop_bounding_box(image, bounding_box), 0.0, (0.0, 0.0)
    
    slope, intercept, _, _ = line_data
    
    # Determine the correct orientation based on head direction
    direction_multiplier = determine_head_direction(keypoints, keypoints_conf, slope, intercept)
    
    # Calculate bounding box center
    x1, y1, x2, y2 = bounding_box
    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0
    
    # Find closest point on regression line to bounding box center
    # Line equation: y = slope * x + intercept
    # Perpendicular line through bbox center: y = (-1/slope) * (x - bbox_center_x) + bbox_center_y
    # Intersection point (closest point on line to bbox center):
    if abs(slope) < 1e-10:  # Nearly horizontal line
        closest_x = bbox_center_x
        closest_y = intercept
    else:
        closest_x = (bbox_center_y - intercept + bbox_center_x / slope) / (slope + 1.0 / slope)
        closest_y = slope * closest_x + intercept
    
    rotation_center = (closest_x, closest_y)
    
    # Calculate rotation angle to make the line horizontal
    # Current line angle from horizontal
    line_angle_rad = np.arctan(slope)
    line_angle_deg = np.degrees(line_angle_rad)
    
    # Apply direction correction and rotate to make line horizontal
    # If direction_multiplier is -1, we need to add 180 degrees to flip orientation
    base_rotation = line_angle_deg
    orientation_correction = 180.0 if direction_multiplier < 0 else 0.0
    rotation_angle_deg = base_rotation + orientation_correction
    
    # Rotate the entire image around the rotation center
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle_deg, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Transform bounding box coordinates to rotated space
    bbox_corners = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ])
    # Add homogeneous coordinate
    bbox_corners_homo = np.column_stack([bbox_corners, np.ones(4)])
    # Transform corners
    rotated_corners = (rotation_matrix @ bbox_corners_homo.T).T
    
    # Find new bounding box in rotated space
    rot_x_min, rot_y_min = np.min(rotated_corners, axis=0)
    rot_x_max, rot_y_max = np.max(rotated_corners, axis=0)
    
    # Calculate square crop size (largest dimension of rotated bbox)
    bbox_width = rot_x_max - rot_x_min
    bbox_height = rot_y_max - rot_y_min
    crop_size = max(bbox_width, bbox_height)
    
    # Center the square crop on the rotated bounding box center
    rot_bbox_center_x = (rot_x_min + rot_x_max) / 2.0
    rot_bbox_center_y = (rot_y_min + rot_y_max) / 2.0
    
    half_crop = crop_size / 2.0
    crop_x1 = int(max(0, rot_bbox_center_x - half_crop))
    crop_y1 = int(max(0, rot_bbox_center_y - half_crop))
    crop_x2 = int(min(w, rot_bbox_center_x + half_crop))
    crop_y2 = int(min(h, rot_bbox_center_y + half_crop))
    
    # Ensure we get a square crop (adjust if we hit image boundaries)
    actual_width = crop_x2 - crop_x1
    actual_height = crop_y2 - crop_y1
    actual_size = min(actual_width, actual_height)
    
    # Re-center the crop to ensure it's square
    crop_x2 = crop_x1 + actual_size
    crop_y2 = crop_y1 + actual_size
    
    # Extract the square crop
    cropped_rotated = rotated_image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_rotated, rotation_angle_deg, rotation_center


def process_and_crop_image(image_path: Path, visual: bool = False, output_path: Path = None) -> np.ndarray:
    """
    Process a single image and return cropped bounding box
    
    Args:
        image_path: Path to input image
        visual: Whether to show visualization plots
        output_path: Path to save the cropped image (optional)
        
    Returns:
        Cropped image as numpy array, or None if no detection
    """
    # Load image
    if visual:
        print(f"Loading image: {image_path}")
    image = mpimg.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    
    orig_h, orig_w = image.shape[0], image.shape[1]
    if visual:
        print(f"Image size: {orig_w}x{orig_h}")
    
    # Run inference with hardcoded defaults
    if visual:
        print("Running inference...")
    boxes, keypoints, keypoints_conf, classes, class_names = inference_single_image(image)
    
    # Hardcoded keypoint names
    keypoint_names = ["left-out", "left-eye", "left-edge", "right-edge", "right-eye", "right-out"]
    
    # Process results
    if boxes is not None and len(boxes) > 0:
        if visual:
            print(f"Found {len(boxes)} detection(s)")
        
        # Use the first detection for cropping
        first_box = boxes[0]
        first_keypoints = keypoints[0] if keypoints is not None else None
        first_keypoints_conf = keypoints_conf[0] if keypoints_conf is not None else None
        
        # Get both regular crop and aligned crop
        cropped_image = crop_bounding_box(image, first_box)
        aligned_crop, rotation_angle, rotation_center = rotate_and_crop_aligned(
            image, first_box, first_keypoints, first_keypoints_conf
        )
        
        # Show visualization if requested
        if visual:
            # Create 4-panel figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))
            
            # Top-left: Original image
            ax1.imshow(image)
            ax1.set_title(f"Original Image ({orig_w}x{orig_h})")
            ax1.axis("off")
            
            # Top-right: Image with inference results and rotation center
            ax2.imshow(image)
            ax2.set_title(f"With Inference Results + Rotation Center")
            ax2.axis("off")
            draw_prediction(
                ax=ax2,
                img_h=orig_h,
                img_w=orig_w,
                boxes_xyxy=boxes,
                classes=classes,
                keypoints_xy=keypoints,
                keypoints_conf=keypoints_conf,
                class_names=class_names,
                keypoint_names=keypoint_names,
            )
            
            # Draw line of best fit for the first detection
            if keypoints is not None and len(keypoints) > 0:
                draw_line_of_best_fit(ax2, first_keypoints, first_keypoints_conf)
            
            # Mark rotation center
            if rotation_center != (0.0, 0.0):
                ax2.scatter([rotation_center[0]], [rotation_center[1]], 
                          c='red', s=100, marker='+', linewidth=4, zorder=6)
                ax2.text(rotation_center[0] + 10, rotation_center[1] + 10, 
                        f'Rotation Center\n({rotation_center[0]:.1f}, {rotation_center[1]:.1f})',
                        color='red', fontsize=10, fontweight='bold')
            
            # Highlight the box we're cropping with a thicker border
            x1, y1, x2, y2 = first_box
            crop_rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor="yellow", facecolor="none", linestyle="--"
            )
            ax2.add_patch(crop_rect)
            ax2.text(x1, y1 - 10, "BBOX", color="yellow", fontsize=12, fontweight="bold")
            
            # Bottom-left: Regular cropped bounding box
            ax3.imshow(cropped_image)
            crop_h, crop_w = cropped_image.shape[:2]
            ax3.set_title(f"Regular Crop ({crop_w}x{crop_h})")
            ax3.axis("off")
            
            # Bottom-right: Aligned and cropped result
            ax4.imshow(aligned_crop)
            aligned_h, aligned_w = aligned_crop.shape[:2]
            ax4.set_title(f"Aligned & Cropped ({aligned_w}x{aligned_h})\nRotated {rotation_angle:.1f}°")
            ax4.axis("off")
            
            plt.tight_layout()
            plt.show()
            
            # Print detection details
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                class_id = classes[i] if classes is not None else -1
                class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "unknown"
                print(f"Detection {i+1}: {class_name} at ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                
                if keypoints is not None and i < len(keypoints):
                    kps = keypoints[i]
                    kps_conf = keypoints_conf[i] if keypoints_conf is not None else None
                    print(f"  Keypoints: {len(kps)} points")
                    for k, (kx, ky) in enumerate(kps):
                        kp_name = keypoint_names[k] if k < len(keypoint_names) else f"kp_{k}"
                        conf_val = kps_conf[k] if kps_conf is not None else 1.0
                        print(f"    {kp_name}: ({kx:.1f}, {ky:.1f}) conf={conf_val:.2f}")
                    
                    # Print midpoints and line of best fit info
                    if len(kps) >= 6:
                        keypoint_pairs = [(5, 0), (4, 1), (3, 2)]
                        midpoints = []
                        conf_threshold = 0.5
                        
                        print(f"  Midpoints used for line fitting:")
                        for idx1, idx2 in keypoint_pairs:
                            # Check confidence
                            if kps_conf is not None:
                                conf1 = kps_conf[idx1] if idx1 < len(kps_conf) else 0.0
                                conf2 = kps_conf[idx2] if idx2 < len(kps_conf) else 0.0
                                if conf1 < conf_threshold or conf2 < conf_threshold:
                                    print(f"    Pair {idx1}&{idx2}: SKIPPED (low confidence)")
                                    continue
                            
                            # Calculate midpoint
                            kp1 = kps[idx1]
                            kp2 = kps[idx2]
                            midpoint = (kp1 + kp2) / 2.0
                            midpoints.append(midpoint)
                            print(f"    Pair {idx1}&{idx2}: midpoint at ({midpoint[0]:.1f}, {midpoint[1]:.1f})")
                        
                        # Print line of best fit info
                        line_data = compute_line_of_best_fit(kps, kps_conf)
                        if line_data is not None:
                            slope, intercept, _, _ = line_data
                            line_angle_deg = np.degrees(np.arctan(slope))
                            print(f"  Line of Best Fit: slope={slope:.3f}, intercept={intercept:.1f}")
                            print(f"    Line angle from horizontal: {line_angle_deg:.1f}° ({len(midpoints)} midpoints)")
            
            # Print rotation information
            if visual and rotation_center != (0.0, 0.0):
                line_data = compute_line_of_best_fit(first_keypoints, first_keypoints_conf)
                if line_data is not None:
                    slope, intercept, _, _ = line_data
                    line_angle_deg = np.degrees(np.arctan(slope))
                    direction_mult = determine_head_direction(first_keypoints, first_keypoints_conf, slope, intercept)
                    orientation_correction = 180.0 if direction_mult < 0 else 0.0
                    
                    print(f"  Rotation: {rotation_angle:.1f}° around ({rotation_center[0]:.1f}, {rotation_center[1]:.1f})")
                    print(f"    Original line angle: {line_angle_deg:.1f}°")
                    print(f"    Direction multiplier: {direction_mult:.1f} (orientation correction: {orientation_correction:.0f}°)")
                    print(f"    Final rotation: {line_angle_deg:.1f}° + {orientation_correction:.0f}° = {rotation_angle:.1f}°")
                print(f"  Aligned crop size: {aligned_crop.shape[1]}x{aligned_crop.shape[0]}")
        
        # Save the aligned crop if output_path is provided
        if output_path is not None:
            # Convert to PIL Image and save
            if aligned_crop.max() <= 1.0:
                # Float image, convert to uint8
                aligned_crop_uint8 = (aligned_crop * 255).astype(np.uint8)
            else:
                aligned_crop_uint8 = aligned_crop.astype(np.uint8)
            
            pil_image = PILImage.fromarray(aligned_crop_uint8)
            pil_image.save(output_path)
            if visual:
                print(f"  Saved aligned crop to: {output_path}")
        
        return aligned_crop  # Return the aligned crop instead of regular crop
        
    else:
        if visual:
            print("No detections found")
            # Show original image only
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(image)
            ax.set_title(f"Original Image - No Detections ({orig_w}x{orig_h})")
            ax.axis("off")
            plt.tight_layout()
            plt.show()
        
        return None


def get_image_files(folder_path: Path) -> list[Path]:
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to folder containing images
        
    Returns:
        List of image file paths
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    for ext in supported_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_folder(folder_path: Path, output_folder: Path, visual: bool = False) -> None:
    """
    Process all images in a folder and save outputs
    
    Args:
        folder_path: Path to input folder
        output_folder: Path to output folder
        visual: Whether to show visualization plots
    """
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    successful_crops = 0
    failed_crops = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
        
        # Generate output filename
        output_filename = f"aligned_crop_{image_path.stem}.jpg"
        output_path = output_folder / output_filename
        
        try:
            cropped_image = process_and_crop_image(
                image_path=image_path,
                visual=visual,
                output_path=output_path
            )
            
            if cropped_image is not None:
                successful_crops += 1
                print(f"  ✓ Saved to: {output_path}")
            else:
                failed_crops += 1
                print(f"  ✗ No detection found - skipped")
                
        except Exception as e:
            failed_crops += 1
            print(f"  ✗ Error processing {image_path.name}: {e}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Successful crops: {successful_crops}")
    print(f"Failed/skipped: {failed_crops}")
    print(f"Output folder: {output_folder}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description="Process single image or folder: run inference and return cropped bounding boxes"
    )
    
    # Create mutually exclusive group for image vs folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )
    input_group.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output folder for cropped images (default: ./output)"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show visualization plots during processing"
    )
    return parser


def main():
    """Main function"""
    args = build_arg_parser().parse_args()
    
    output_folder = Path(args.output)
    
    if args.image:
        # Process single image
        image_path = Path(args.image)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # For single image, save to output folder if specified
        if args.output != "./output" or args.folder:  # Save if custom output or processing folder
            output_folder.mkdir(parents=True, exist_ok=True)
            output_filename = f"aligned_crop_{image_path.stem}.jpg"
            output_path = output_folder / output_filename
        else:
            output_path = None
        
        cropped_image = process_and_crop_image(
            image_path=image_path,
            visual=args.visual,
            output_path=output_path
        )
        
        if cropped_image is not None:
            if output_path:
                print(f"Image processed successfully. Saved to: {output_path}")
            else:
                print("Image processed successfully. Cropped image returned.")
        else:
            print("No detection found - no crop returned.")
    
    elif args.folder:
        # Process folder
        folder_path = Path(args.folder)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")
        
        process_folder(
            folder_path=folder_path,
            output_folder=output_folder,
            visual=args.visual
        )


if __name__ == "__main__":
    main()

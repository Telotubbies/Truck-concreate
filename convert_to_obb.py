"""
Convert Regular YOLO Labels to OBB Format
Uses YOLO-World to detect oriented bounding boxes with angles
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def estimate_angle_from_bbox(img, bbox):
    """
    Estimate rotation angle from bounding box region
    
    Args:
        img: Input image
        bbox: Bounding box [x_center, y_center, width, height] (normalized)
        
    Returns:
        angle: Rotation angle in radians
    """
    h, w = img.shape[:2]
    
    # Convert normalized to pixel coordinates
    x_center = int(bbox[0] * w)
    y_center = int(bbox[1] * h)
    box_w = int(bbox[2] * w)
    box_h = int(bbox[3] * h)
    
    # Extract ROI
    x1 = max(0, x_center - box_w // 2)
    y1 = max(0, y_center - box_h // 2)
    x2 = min(w, x_center + box_w // 2)
    y2 = min(h, y_center + box_h // 2)
    
    roi = img[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit minimum area rectangle
    if len(largest_contour) >= 5:
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]  # Angle in degrees
        
        # Convert to radians and normalize to [-pi/2, pi/2]
        angle_rad = np.deg2rad(angle)
        
        # Normalize angle
        if angle_rad > np.pi / 2:
            angle_rad -= np.pi
        elif angle_rad < -np.pi / 2:
            angle_rad += np.pi
            
        return angle_rad
    
    return 0.0


def convert_label_to_obb(label_path, image_path, output_label_path):
    """
    Convert regular YOLO label to OBB format
    
    Args:
        label_path: Path to input label file
        image_path: Path to corresponding image
        output_label_path: Path to output OBB label file
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        # Skip silently for files with encoding issues
        return False
    
    # Read labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    obb_lines = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Estimate angle for mixer_drum (class 1)
        if class_id == 1:
            angle = estimate_angle_from_bbox(img, [x_center, y_center, width, height])
        else:
            # For trucks (class 0), assume no rotation
            angle = 0.0
        
        # OBB format: class x_center y_center width height angle
        obb_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {angle:.6f}\n"
        obb_lines.append(obb_line)
    
    # Write OBB labels
    with open(output_label_path, 'w') as f:
        f.writelines(obb_lines)
    
    return True


def convert_dataset_to_obb(dataset_dir, output_dir):
    """
    Convert entire dataset to OBB format
    
    Args:
        dataset_dir: Input dataset directory
        output_dir: Output dataset directory
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    print("="*70)
    print("CONVERTING DATASET TO OBB FORMAT")
    print("="*70)
    print(f"Input: {dataset_path}")
    print(f"Output: {output_path}")
    
    # Create output directories
    for split in ['train', 'valid']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    total_converted = 0
    total_failed = 0
    
    # Process each split
    for split in ['train', 'valid']:
        print(f"\n[{split.upper()}] Processing...")
        
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'
        
        if not img_dir.exists():
            print(f"  [SKIP] No images found in {img_dir}")
            continue
        
        # Get all images
        image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for img_file in tqdm(image_files, desc=f"  Converting {split}"):
            # Copy image
            output_img = output_path / split / 'images' / img_file.name
            shutil.copy(img_file, output_img)
            
            # Convert label
            label_file = lbl_dir / f"{img_file.stem}.txt"
            output_label = output_path / split / 'labels' / f"{img_file.stem}.txt"
            
            if label_file.exists():
                success = convert_label_to_obb(label_file, img_file, output_label)
                if success:
                    total_converted += 1
                else:
                    total_failed += 1
            else:
                # No label file, create empty
                output_label.touch()
    
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Total converted: {total_converted}")
    print(f"Total failed: {total_failed}")
    print(f"Output directory: {output_path}")
    print("="*70)
    
    # Copy data.yaml if exists
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        shutil.copy(data_yaml, output_path / 'data.yaml')
        print(f"\n[OK] Copied data.yaml")
    
    return output_path


def visualize_obb(image_path, label_path, output_path):
    """
    Visualize OBB labels on image
    
    Args:
        image_path: Path to image
        label_path: Path to OBB label file
        output_path: Path to save visualization
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    # Read OBB labels
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        width = float(parts[3]) * w
        height = float(parts[4]) * h
        angle = float(parts[5])
        
        # Create rotated rectangle
        rect = ((x_center, y_center), (width, height), np.rad2deg(angle))
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        # Draw
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
        cv2.drawContours(img, [box], 0, color, 2)
        
        # Draw angle indicator
        angle_line_length = min(width, height) / 2
        end_x = int(x_center + angle_line_length * np.cos(angle))
        end_y = int(y_center + angle_line_length * np.sin(angle))
        cv2.line(img, (int(x_center), int(y_center)), (end_x, end_y), (255, 0, 0), 2)
        
        # Label
        label = f"{'truck' if class_id == 0 else 'drum'} {np.rad2deg(angle):.1f}°"
        cv2.putText(img, label, (int(x_center), int(y_center) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(str(output_path), img)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to OBB format')
    parser.add_argument('--input', type=str, default='./merged_dataset',
                       help='Input dataset directory')
    parser.add_argument('--output', type=str, default='./merged_dataset_obb',
                       help='Output dataset directory')
    parser.add_argument('--visualize', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Convert dataset
    output_dir = convert_dataset_to_obb(args.input, args.output)
    
    # Visualize samples
    if args.visualize > 0:
        print(f"\n[VIS] Creating {args.visualize} visualization samples...")
        
        vis_dir = Path(args.output) / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        train_images = list((Path(args.output) / 'train' / 'images').glob('*.jpg'))[:args.visualize]
        
        for i, img_file in enumerate(train_images):
            label_file = Path(args.output) / 'train' / 'labels' / f"{img_file.stem}.txt"
            output_file = vis_dir / f"sample_{i+1}.jpg"
            
            if label_file.exists():
                visualize_obb(img_file, label_file, output_file)
        
        print(f"[OK] Visualizations saved to: {vis_dir}")
    
    print(f"\n✅ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Check visualizations in: {args.output}/visualizations/")
    print(f"  2. Update config.yaml:")
    print(f"     dataset:")
    print(f"       merged_dir: \"{args.output}\"")
    print(f"  3. Run training:")
    print(f"     python main.py --mode train --config config.yaml")

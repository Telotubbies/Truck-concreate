"""
Demo: Rotation Detection on Video
Tests the complete rotation detection pipeline
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import get_config
from rotation_detector import RotationDetector
from video_processor import VideoProcessor

def create_demo_video():
    """Create a simple demo video with rotating circle"""
    print("Creating demo video with rotating pattern...")
    
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_rotation.mp4', fourcc, fps, (width, height))
    
    for i in range(frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw rotating pattern
        center = (width // 2, height // 2)
        radius = 100
        angle = (i / frames) * 360 * 2  # 2 full rotations
        
        # Draw circle
        cv2.circle(frame, center, radius, (100, 100, 100), -1)
        
        # Draw rotating line
        end_x = int(center[0] + radius * np.cos(np.radians(angle)))
        end_y = int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.line(frame, center, (end_x, end_y), (0, 255, 0), 3)
        
        # Add text
        cv2.putText(frame, f"Angle: {angle:.1f}°", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("[OK] Demo video created: demo_rotation.mp4")
    return 'demo_rotation.mp4'


def test_rotation_detector():
    """Test rotation detector on demo video"""
    print("\n" + "="*70)
    print("ROTATION DETECTION TEST")
    print("="*70)
    
    # Load config
    config = get_config(yaml_path='config.yaml')
    
    # Create demo video
    video_path = create_demo_video()
    
    # Initialize rotation detector
    print("\n[INIT] Rotation Detector")
    detector = RotationDetector(config['rotation_detection'])
    
    # Process video
    print("\n[PROCESS] Analyzing rotation...")
    cap = cv2.VideoCapture(video_path)
    
    prev_frame = None
    frame_count = 0
    rotation_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect rotation
        if prev_frame is not None:
            is_rotating, rotation_info = detector.detect_rotation(
                prev_frame, frame, frame_count
            )
            
            if is_rotating:
                rotation_detected += 1
                print(f"  Frame {frame_count}: ROTATING! "
                      f"(Flow: {rotation_info.get('flow_magnitude', 0):.2f}, "
                      f"SSIM: {rotation_info.get('ssim', 0):.3f})")
        
        prev_frame = frame.copy()
    
    cap.release()
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total frames: {frame_count}")
    print(f"Rotation detected: {rotation_detected} frames")
    print(f"Detection rate: {rotation_detected/frame_count*100:.1f}%")
    print("="*70)
    
    return rotation_detected > 0


def test_video_processor():
    """Test complete video processor with state machine"""
    print("\n" + "="*70)
    print("VIDEO PROCESSOR TEST (with State Machine)")
    print("="*70)
    
    # Load config
    config = get_config(yaml_path='config.yaml')
    
    # Use existing model or create dummy
    model_path = 'runs/detect/runs/obb/mixer_truck_yolo26n2/weights/best.pt'
    if not Path(model_path).exists():
        print("[WARNING] No trained model found, using pretrained yolo26n.pt")
        model_path = 'yolo26n.pt'
    
    # Initialize processor
    print(f"\n[INIT] Video Processor (model: {model_path})")
    processor = VideoProcessor(
        model_path=model_path,
        rotation_config=config['rotation_detection'],
        video_config=config['video_processing']
    )
    
    # Create demo video
    video_path = create_demo_video()
    
    # Process video
    print("\n[PROCESS] Processing video with state machine...")
    output_path = 'output_rotation_demo.mp4'
    
    results = processor.process_video(
        video_path=video_path,
        output_path=output_path,
        show_progress=True
    )
    
    # Display results
    print("\n" + "="*70)
    print("VIDEO PROCESSING RESULTS")
    print("="*70)
    print(f"Total frames: {results['total_frames']}")
    print(f"Detections: {results['total_detections']}")
    print(f"\nState Distribution:")
    for state, count in results['state_counts'].items():
        percentage = count / results['total_frames'] * 100
        print(f"  {state}: {count} frames ({percentage:.1f}%)")
    
    print(f"\nOutput saved: {output_path}")
    print("="*70)
    
    return True


def main():
    """Run all rotation detection tests"""
    print("="*70)
    print("CONCRETE MIXER ROTATION DETECTION - DEMO")
    print("="*70)
    
    # Test 1: Basic rotation detector
    print("\n[TEST 1] Basic Rotation Detector")
    success1 = test_rotation_detector()
    
    # Test 2: Complete video processor
    print("\n[TEST 2] Complete Video Processor")
    success2 = test_video_processor()
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"✓ Rotation Detector: {'PASS' if success1 else 'FAIL'}")
    print(f"✓ Video Processor: {'PASS' if success2 else 'FAIL'}")
    print("\nGenerated files:")
    print("  - demo_rotation.mp4 (input)")
    print("  - output_rotation_demo.mp4 (output with annotations)")
    print("\nNext steps:")
    print("  1. Test on real concrete mixer video")
    print("  2. Adjust thresholds in config.yaml")
    print("  3. Fine-tune state machine transitions")
    print("="*70)


if __name__ == "__main__":
    main()

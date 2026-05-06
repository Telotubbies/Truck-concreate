#!/usr/bin/env python3
"""
Main Entry Point for Concrete Mixer Truck Detection System
Production-ready pipeline using YOLO26n-OBB
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import get_config
from data_loader import DataLoader
from trainer import ModelTrainer
from video_processor import VideoProcessor


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Concrete Mixer Truck Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (data + train + validate)
  python main.py --mode full

  # Use custom YAML config
  python main.py --mode full --config config.yaml

  # Data preparation only
  python main.py --mode data

  # Training only (requires prepared data)
  python main.py --mode train

  # Video processing (requires trained model)
  python main.py --mode video --input video.mp4 --output result.mp4

  # Validation only
  python main.py --mode validate --model runs/obb/mixer_truck_yolo26n/weights/best.pt
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['full', 'data', 'train', 'validate', 'video'],
                       help='Execution mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (optional)')
    parser.add_argument('--input', type=str, help='Input video path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video path (for video mode)')
    parser.add_argument('--model', type=str, help='Model path (for validate/video mode)')
    
    args = parser.parse_args()
    
    # Load configuration (with optional YAML override)
    config = get_config(yaml_path=args.config)
    
    print("="*70)
    print(f"CONCRETE MIXER TRUCK DETECTION SYSTEM")
    print(f"Version: {config['project'].VERSION}")
    print(f"Model: {config['model'].MODEL_NAME}")
    print("="*70)
    
    # ========================================================================
    # MODE: DATA PREPARATION
    # ========================================================================
    if args.mode in ['full', 'data']:
        print("\n[MODE] Data Preparation")
        
        loader = DataLoader(config['dataset'])
        
        # Load datasets
        dataset1, dataset2 = loader.load_all_datasets()
        
        if not dataset1 and not dataset2:
            print("[ERROR] No datasets available!")
            return 1
        
        # Merge datasets
        merged_dir = loader.merge_datasets(
            [dataset1, dataset2],
            config['dataset'].MERGED_DIR
        )
        
        # Create data.yaml
        data_yaml_path = loader.create_data_yaml(
            merged_dir,
            config['dataset'].CLASSES,
            config['dataset'].NUM_CLASSES
        )
        
        print(f"\n[OK] Data preparation complete!")
        print(f"     Data YAML: {data_yaml_path}")
        
        if args.mode == 'data':
            return 0
    
    # ========================================================================
    # MODE: TRAINING
    # ========================================================================
    if args.mode in ['full', 'train']:
        print("\n[MODE] Training")
        
        # Check data.yaml exists
        data_yaml_path = Path(config['dataset'].MERGED_DIR) / 'data.yaml'
        if not data_yaml_path.exists():
            print(f"[ERROR] data.yaml not found: {data_yaml_path}")
            print("        Run with --mode data first")
            return 1
        
        # Initialize trainer
        trainer = ModelTrainer(
            config['model'],
            config['mlflow'],
            config['paths']
        )
        
        # Setup MLflow
        trainer.setup_mlflow()
        
        # Load model
        model = trainer.load_model()
        if not model:
            print("[ERROR] Failed to load model")
            return 1
        
        # Train
        best_model_path, metrics = trainer.train(str(data_yaml_path))
        
        print(f"\n[OK] Training complete!")
        print(f"     Best model: {best_model_path}")
        
        if args.mode == 'train':
            return 0
    
    # ========================================================================
    # MODE: VALIDATION
    # ========================================================================
    if args.mode == 'validate':
        print("\n[MODE] Validation")
        
        model_path = args.model or f"{config['paths'].RUNS_DIR}/{config['paths'].MODEL_NAME}/weights/best.pt"
        data_yaml_path = Path(config['dataset'].MERGED_DIR) / 'data.yaml'
        
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
            return 1
        
        if not data_yaml_path.exists():
            print(f"[ERROR] data.yaml not found: {data_yaml_path}")
            return 1
        
        trainer = ModelTrainer(
            config['model'],
            config['mlflow'],
            config['paths']
        )
        
        val_results = trainer.validate(model_path, str(data_yaml_path))
        
        print(f"\n[OK] Validation complete!")
        return 0
    
    # ========================================================================
    # MODE: VIDEO PROCESSING
    # ========================================================================
    if args.mode == 'video':
        print("\n[MODE] Video Processing")
        
        if not args.input:
            print("[ERROR] --input required for video mode")
            return 1
        
        model_path = args.model or f"{config['paths'].RUNS_DIR}/{config['paths'].MODEL_NAME}/weights/best.pt"
        output_path = args.output or 'output_analysis.mp4'
        
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
            print("        Train model first or specify --model path")
            return 1
        
        if not Path(args.input).exists():
            print(f"[ERROR] Input video not found: {args.input}")
            return 1
        
        processor = VideoProcessor(model_path, config['rotation'])
        stats = processor.process_video(args.input, output_path)
        
        if stats:
            print(f"\n[OK] Video processing complete!")
            return 0
        else:
            print(f"\n[ERROR] Video processing failed!")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

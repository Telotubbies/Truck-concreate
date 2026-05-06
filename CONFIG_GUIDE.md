# 📝 Configuration Guide

## Overview

The system supports **two configuration methods**:

1. **Python Dataclasses** (Default) - `src/config.py`
2. **YAML Configuration** (Recommended) - `config.yaml`

YAML configuration provides easier parameter tuning without modifying code.

---

## 🎯 Quick Start

### Using Default Configuration

```bash
python main.py --mode full
```

### Using YAML Configuration

```bash
python main.py --mode full --config config.yaml
```

---

## 📋 Configuration Sections

### 1. Project Information

```yaml
project:
  name: "concrete_mixer_obb_detection"
  version: "1.0.0"
  author: "Senior Computer Vision Engineer"
```

### 2. Dataset Configuration

```yaml
dataset:
  data_dir: "./data"
  merged_dir: "./merged_dataset"
  
  # Dataset names
  dataset_1_name: "Mixer.yolo26"
  dataset_2_name: "concrete mixed truck.yolo26"
  
  # Classes
  num_classes: 2
  classes:
    0: "truck"
    1: "mixer_drum"
```

### 3. Model Configuration

```yaml
model:
  name: "yolo26n-obb.pt"      # Model file
  type: "OBB"                  # Oriented Bounding Box
  variant: "nano"              # Nano variant
  
  input_size: 640              # Input image size
  batch_size: 16               # Training batch size
  epochs: 100                  # Number of epochs
  patience: 20                 # Early stopping patience
  
  # Optimizer
  optimizer: "SGD"
  learning_rate: 0.01
  momentum: 0.937
```

### 4. Data Augmentation

```yaml
augmentation:
  # Geometric augmentations
  degrees: 45.0                # Rotation ±45°
  translate: 0.1               # Translation
  scale: 0.5                   # Scale range
  perspective: 0.001           # Perspective distortion
  
  # Flip augmentations
  flipud: 0.0                  # Vertical flip
  fliplr: 0.5                  # Horizontal flip (50%)
  
  # Advanced augmentations
  mosaic: 1.0                  # Mosaic (100%)
  mixup: 0.1                   # MixUp (10%)
  
  # Color augmentations
  hsv_h: 0.015                 # Hue
  hsv_s: 0.7                   # Saturation
  hsv_v: 0.4                   # Value
```

### 5. Rotation Detection

```yaml
rotation_detection:
  # Optical Flow (Farneback)
  optical_flow:
    enabled: true
    magnitude_threshold: 2.0   # Minimum flow for rotation
    pyr_scale: 0.5
    levels: 3
    winsize: 15
    iterations: 3
  
  # SSIM (Structural Similarity)
  ssim:
    enabled: true
    threshold: 0.85            # Lower = more change detected
    pixel_diff_threshold: 0.15
    pixel_diff_intensity: 30
```

### 6. State Machine

```yaml
state_machine:
  velocity_threshold: 5.0      # Pixels/frame for "moving"
  position_history_size: 10    # Position tracking
  state_history_size: 5        # State smoothing
  
  states:
    - name: "POURING_CONCRETE"
      truck_moving: false
      drum_rotating: true
      color: [0, 0, 255]       # Red (BGR)
    
    - name: "IN_TRANSIT"
      truck_moving: true
      drum_rotating: true
      color: [0, 255, 255]     # Yellow (BGR)
    
    - name: "IDLE_WAITING"
      truck_moving: false
      drum_rotating: false
      color: [0, 255, 0]       # Green (BGR)
```

### 7. MLflow Tracking

```yaml
mlflow:
  tracking_dir: "./mlruns"
  experiment_name: "concrete_mixer_obb_yolo26n"
  run_name_prefix: "yolo26n_obb"
  
  log_models: true
  log_artifacts: true
  
  tags:
    model_architecture: "YOLO26n-OBB"
    task: "concrete_mixer_detection"
```

### 8. Video Processing

```yaml
video_processing:
  default_output_name: "output_analysis.mp4"
  codec: "mp4v"
  
  # Visualization
  show_bbox: true
  show_labels: true
  show_confidence: true
  show_state: true
  show_rotation_status: true
  
  # Text overlay
  font_scale: 1.0
  font_thickness: 2
  font_color: [255, 255, 255]  # White (BGR)
```

### 9. Inference Configuration

```yaml
inference:
  conf_threshold: 0.25         # Confidence threshold
  iou_threshold: 0.45          # NMS IoU threshold
  max_det: 300                 # Max detections per image
  
  device: "cuda:0"             # cuda:0, cpu, or mps
  half_precision: false        # Use FP16
```

---

## 🔧 Common Configuration Tasks

### Change Training Parameters

Edit `config.yaml`:

```yaml
model:
  batch_size: 32               # Increase batch size
  epochs: 200                  # More epochs
  learning_rate: 0.02          # Higher learning rate
```

### Adjust Augmentation Strength

```yaml
augmentation:
  degrees: 30.0                # Less rotation
  mosaic: 0.5                  # 50% mosaic
  mixup: 0.0                   # Disable mixup
```

### Tune Rotation Detection

```yaml
rotation_detection:
  optical_flow:
    magnitude_threshold: 3.0   # More sensitive
  
  ssim:
    threshold: 0.90            # Less sensitive
```

### Change State Machine Thresholds

```yaml
state_machine:
  velocity_threshold: 10.0     # Higher threshold for "moving"
  position_history_size: 20    # More history
```

---

## 📊 Configuration Priority

When both Python and YAML configs exist:

1. **Python defaults** are loaded first
2. **YAML values** override Python defaults
3. **Command-line arguments** override everything

Example:
```bash
# Uses YAML config
python main.py --mode train --config config.yaml

# Uses Python defaults
python main.py --mode train

# Override model path
python main.py --mode validate --model custom_model.pt
```

---

## 🎨 Creating Custom Configurations

### 1. Copy Template

```bash
cp config.yaml my_config.yaml
```

### 2. Edit Parameters

```yaml
# my_config.yaml
model:
  batch_size: 8                # For smaller GPU
  epochs: 50                   # Quick training
```

### 3. Use Custom Config

```bash
python main.py --mode train --config my_config.yaml
```

---

## 📝 Configuration Examples

### Example 1: Fast Training (Testing)

```yaml
model:
  batch_size: 32
  epochs: 10
  patience: 5

augmentation:
  mosaic: 0.0
  mixup: 0.0
```

### Example 2: High Accuracy (Production)

```yaml
model:
  batch_size: 16
  epochs: 300
  patience: 50

augmentation:
  degrees: 60.0
  mosaic: 1.0
  mixup: 0.2
```

### Example 3: Edge Deployment (Low Resource)

```yaml
model:
  batch_size: 8
  input_size: 416              # Smaller input

inference:
  half_precision: true         # FP16
  max_det: 100                 # Fewer detections
```

---

## 🔍 Validation

### Check Current Configuration

```python
from src.config import get_config

# Load and print
config = get_config(yaml_path="config.yaml")
print(f"Batch size: {config['model'].BATCH_SIZE}")
print(f"Epochs: {config['model'].EPOCHS}")
```

### Export Configuration

```python
from src.config import get_config, save_config_to_yaml

config = get_config()
save_config_to_yaml(config, "config_backup.yaml")
```

---

## ⚠️ Important Notes

1. **YAML Syntax**: Use spaces, not tabs
2. **Boolean Values**: `true`/`false` (lowercase)
3. **Colors**: BGR format `[B, G, R]`
4. **Paths**: Use forward slashes `/` or double backslashes `\\`

---

## 🐛 Troubleshooting

### YAML Not Loading

```
[WARNING] YAML config not found: config.yaml
          Using default Python configuration
```

**Solution:** Check file path and name

### Invalid YAML Syntax

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution:** Check indentation (use spaces, not tabs)

### Parameter Not Applied

**Solution:** Make sure parameter name matches exactly in YAML

---

## 📚 Full Parameter Reference

See `config.yaml` for complete list of all available parameters with descriptions.

**Total Sections:** 14
**Total Parameters:** 150+

---

## 🔗 Related Files

- `config.yaml` - Main YAML configuration
- `src/config.py` - Python configuration classes
- `main.py` - Main entry point with config loading
- `README_PRODUCTION.md` - Production guide

---

**Last Updated:** 2026-04-14

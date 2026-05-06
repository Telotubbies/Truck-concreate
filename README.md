# 🚛 Concrete Mixer Truck Detection System
## Production-Ready YOLO26n-OBB Implementation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-26n--OBB-green)](https://github.com/ultralytics/ultralytics)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)](https://mlflow.org/)

---

## 📋 Overview

Real-time detection and state classification of concrete mixer trucks from CCTV footage using YOLO26n-OBB with rotation detection.

**Key Features:**
- 🎯 Oriented Bounding Box (OBB) detection
- 🔄 Dual rotation detection (Optical Flow + SSIM)
- 🎛️ State machine (POURING/IN_TRANSIT/IDLE)
- 📊 MLflow experiment tracking
- 🌐 Production-ready modular architecture

---

## 📁 Project Structure

```
Truck-concreate/
├── src/                          # Source code modules
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Dataset loading and fusion
│   ├── trainer.py                # Model training with MLflow
│   ├── rotation_detector.py     # Rotation detection logic
│   └── video_processor.py       # Video processing pipeline
│
├── data/                         # Dataset directory
│   ├── Mixer.yolo26.zip         # Dataset 1 (736 images)
│   └── concrete mixed truck.yolo26.zip  # Dataset 2 (74 images)
│
├── main.py                       # Main entry point
│
├── run_full_pipeline.sh          # Run complete pipeline
├── run_data_prep.sh              # Data preparation only
├── run_train.sh                  # Training only
├── run_validate.sh               # Validation only
├── run_video.sh                  # Video processing
├── run_mlflow_ui.sh              # Start MLflow UI
│
├── requirements.txt              # Python dependencies
└── README_PRODUCTION.md          # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd "C:\ปี3\Truck detection\Truck-concreate"

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Extract and merge datasets
bash run_data_prep.sh
```

**Output:**
- `merged_dataset/` - Combined dataset
- `merged_dataset/data.yaml` - YOLO configuration

### 3. Train Model

```bash
# Train YOLO26n-OBB with MLflow tracking
bash run_train.sh
```

**Output:**
- `runs/obb/mixer_truck_yolo26n/weights/best.pt` - Best model
- `mlruns/` - MLflow experiment data

### 4. View Training Results

```bash
# Start MLflow UI
bash run_mlflow_ui.sh

# Open browser: http://localhost:5000
```

### 5. Validate Model

```bash
# Validate on test set
bash run_validate.sh

# Or specify custom model
bash run_validate.sh path/to/model.pt
```

### 6. Process Video

```bash
# Process video with rotation detection
bash run_video.sh input.mp4 output.mp4

# Or use default output name
bash run_video.sh input.mp4
```

---

## 🔧 Advanced Usage

### Full Pipeline (One Command)

```bash
# Run everything: data prep + training + validation
bash run_full_pipeline.sh
```

### Python API Usage

```python
from src.config import get_config
from src.data_loader import DataLoader
from src.trainer import ModelTrainer
from src.video_processor import VideoProcessor

# Load configuration
config = get_config()

# 1. Data preparation
loader = DataLoader(config['dataset'])
dataset1, dataset2 = loader.load_all_datasets()
merged_dir = loader.merge_datasets([dataset1, dataset2], './merged_dataset')

# 2. Training
trainer = ModelTrainer(config['model'], config['mlflow'], config['paths'])
trainer.setup_mlflow()
trainer.load_model()
trainer.train('./merged_dataset/data.yaml')

# 3. Video processing
processor = VideoProcessor('runs/obb/mixer_truck_yolo26n/weights/best.pt', config['rotation'])
stats = processor.process_video('input.mp4', 'output.mp4')
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
@dataclass
class ModelConfig:
    MODEL_NAME: str = "yolo26n-obb.pt"  # Model to use
    BATCH_SIZE: int = 16                 # Training batch size
    EPOCHS: int = 100                    # Training epochs
    DEGREES: float = 45.0                # Rotation augmentation
    # ... more parameters
```

---

## 📊 Model Performance

**Dataset:**
- Training images: 810 (736 + 74)
- Classes: `truck (0)`, `mixer_drum (1)`

**Augmentation Strategy:**
- Rotation: ±45° (multi-viewpoint robustness)
- Perspective: 0.001 (oblique views)
- Mosaic: 1.0 (multi-object learning)

**Expected Performance:**
- mAP50: >0.85
- mAP50-95: >0.60

---

## 🔄 Rotation Detection

### Methods

1. **Optical Flow (Farneback)**
   - Detects pixel motion between frames
   - Threshold: 2.0 pixels mean magnitude
   - Robust to lighting changes

2. **SSIM Temporal Difference**
   - Structural similarity comparison
   - Threshold: SSIM < 0.85
   - Computationally lighter

### State Machine

| State | Truck | Drum | Description |
|-------|-------|------|-------------|
| POURING_CONCRETE | Stopped | Rotating | Actively pouring |
| IN_TRANSIT | Moving | Rotating | Traveling to site |
| IDLE_WAITING | Stopped | Still | Waiting/parked |
| UNKNOWN | - | - | Ambiguous state |

---

## 🐛 Troubleshooting

### Model not found error

```bash
# Make sure yolo26n-obb.pt is available
# Or download it first:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26n-obb.pt
```

### Dataset not found

```bash
# Ensure zip files are in data/ folder:
ls data/
# Should show:
#   Mixer.yolo26.zip
#   concrete mixed truck.yolo26.zip
```

### CUDA out of memory

```python
# Reduce batch size in src/config.py:
BATCH_SIZE: int = 8  # Instead of 16
```

---

## 📝 Shell Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_full_pipeline.sh` | Complete pipeline | `bash run_full_pipeline.sh` |
| `run_data_prep.sh` | Data preparation | `bash run_data_prep.sh` |
| `run_train.sh` | Training | `bash run_train.sh` |
| `run_validate.sh` | Validation | `bash run_validate.sh [model.pt]` |
| `run_video.sh` | Video processing | `bash run_video.sh input.mp4 [output.mp4]` |
| `run_mlflow_ui.sh` | MLflow UI | `bash run_mlflow_ui.sh` |

---

## 🔬 Technical Details

**Model:** YOLO26n-OBB
- Architecture: Nano variant for edge deployment
- Input size: 640x640
- Format: Oriented Bounding Boxes (xywhr)

**Frameworks:**
- Ultralytics YOLO
- PyTorch
- MLflow
- OpenCV

**Rotation Detection:**
- Farneback Optical Flow
- SSIM (Structural Similarity Index)
- State machine with history smoothing

---

## 📄 License

MIT License - See LICENSE file

---

## 👤 Author

Senior Computer Vision Engineer

---

## 🙏 Acknowledgments

- Ultralytics YOLO team
- MLflow community
- Dataset contributors

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review MLflow logs
3. Check console output for detailed errors

---

**Version:** 1.0.0  
**Last Updated:** 2026-04-14

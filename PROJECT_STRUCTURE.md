# 📁 Project Structure

## Overview

Clean, organized, production-ready structure for Concrete Mixer Truck Detection System.

---

## 📂 Directory Structure

```
Truck-concreate/
│
├── 📁 src/                          # Source code modules
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration management
│   ├── data_loader.py               # Dataset loading & fusion
│   ├── trainer.py                   # Model training with MLflow
│   ├── rotation_detector.py         # Rotation detection logic
│   └── video_processor.py           # Video processing pipeline
│
├── 📁 data/                         # Dataset directory
│   ├── Mixer.yolo26.zip             # Dataset 1 (736 images)
│   ├── Mixer.yolo26/                # Extracted dataset 1
│   ├── concrete mixed truck.yolo26.zip  # Dataset 2 (74 images)
│   └── concrete mixed truck.yolo26/ # Extracted dataset 2
│
├── 🐍 main.py                       # Main entry point
├── ⚙️ config.yaml                   # YAML configuration (150+ params)
│
├── 🔧 setup.sh                      # Environment setup
├── 🚀 run_full_pipeline.sh          # Complete pipeline
├── 📊 run_data_prep.sh              # Data preparation
├── 🎓 run_train.sh                  # Model training
├── ✅ run_validate.sh               # Model validation
├── 🎬 run_video.sh                  # Video processing
├── 📈 run_mlflow_ui.sh              # MLflow UI
│
├── 📖 README.md                     # Main documentation
├── 📝 CONFIG_GUIDE.md               # Configuration guide
├── 📋 PROJECT_STRUCTURE.md          # This file
├── 📦 requirements.txt              # Python dependencies
├── 🚫 .gitignore                    # Git ignore patterns
│
└── 📓 Truck-cement.ipynb            # Jupyter notebook (reference)
```

---

## 📄 File Descriptions

### Core Python Modules (`src/`)

| File | Purpose | Lines | Key Functions |
|------|---------|-------|---------------|
| `config.py` | Configuration management | 265 | `get_config()`, `load_yaml_config()` |
| `data_loader.py` | Dataset operations | 180 | `load_dataset()`, `merge_datasets()` |
| `trainer.py` | Model training | 200 | `train()`, `validate()` |
| `rotation_detector.py` | Rotation detection | 250 | `detect_optical_flow()`, `detect_temporal_diff()` |
| `video_processor.py` | Video processing | 150 | `process_video()` |

### Shell Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.sh` | Install dependencies | `bash setup.sh` |
| `run_full_pipeline.sh` | Complete workflow | `bash run_full_pipeline.sh` |
| `run_data_prep.sh` | Prepare datasets | `bash run_data_prep.sh` |
| `run_train.sh` | Train model | `bash run_train.sh` |
| `run_validate.sh` | Validate model | `bash run_validate.sh [model.pt]` |
| `run_video.sh` | Process video | `bash run_video.sh input.mp4` |
| `run_mlflow_ui.sh` | Start MLflow UI | `bash run_mlflow_ui.sh` |

### Documentation

| File | Purpose | Size |
|------|---------|------|
| `README.md` | Main documentation | 7 KB |
| `CONFIG_GUIDE.md` | Configuration guide | 8 KB |
| `PROJECT_STRUCTURE.md` | This file | 5 KB |

### Configuration

| File | Purpose | Parameters |
|------|---------|------------|
| `config.yaml` | YAML configuration | 150+ |
| `requirements.txt` | Python dependencies | 20 packages |
| `.gitignore` | Git ignore patterns | 100+ patterns |

---

## 🗂️ Generated Directories (Not in Git)

These directories are created during execution:

```
Truck-concreate/
├── merged_dataset/          # Merged training data
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── runs/                    # Training outputs
│   └── obb/
│       └── mixer_truck_yolo26n/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.csv
│
├── mlruns/                  # MLflow tracking
│   └── [experiment_id]/
│       └── [run_id]/
│
├── logs/                    # Log files
├── models/                  # Saved models
├── deployment/              # Deployment artifacts
└── cache/                   # Cache files
```

---

## 📊 File Statistics

### Total Files: 16 main files

**Python Files:** 6 (src/)
**Shell Scripts:** 7 (.sh)
**Documentation:** 3 (.md)

### Lines of Code

| Category | Lines |
|----------|-------|
| Python modules | ~1,200 |
| Main script | ~200 |
| Shell scripts | ~200 |
| Documentation | ~500 |
| Configuration | ~450 |
| **Total** | **~2,550** |

---

## 🔄 Workflow

```
1. Setup
   └─> bash setup.sh

2. Data Preparation
   └─> bash run_data_prep.sh
       └─> Loads datasets
       └─> Merges data
       └─> Creates data.yaml

3. Training
   └─> bash run_train.sh
       └─> Trains YOLO26n-OBB
       └─> Logs to MLflow
       └─> Saves best model

4. Validation
   └─> bash run_validate.sh
       └─> Evaluates model
       └─> Reports metrics

5. Video Processing
   └─> bash run_video.sh input.mp4
       └─> Detects trucks
       └─> Analyzes rotation
       └─> Classifies states
       └─> Saves output
```

---

## 🎯 Design Principles

### 1. Separation of Concerns
- Each module has single responsibility
- Clear interfaces between components

### 2. Configuration Management
- Centralized in `config.yaml`
- Override-able via command line
- Type-safe with validation

### 3. Production Ready
- Error handling throughout
- Logging and monitoring
- Reproducible results

### 4. Easy to Use
- Shell scripts for common tasks
- Clear documentation
- Intuitive structure

### 5. Maintainable
- Modular design
- Clean code
- Comprehensive comments

---

## 🚀 Quick Commands

```bash
# Setup environment
bash setup.sh

# Full pipeline
bash run_full_pipeline.sh

# Custom config
python main.py --mode full --config my_config.yaml

# Process video
bash run_video.sh input.mp4 output.mp4

# View results
bash run_mlflow_ui.sh
```

---

## 📝 Notes

- All shell scripts are executable
- Python 3.10+ required
- CUDA recommended for training
- MLflow UI runs on port 5000

---

## 🔗 Related Documentation

- [README.md](README.md) - Main documentation
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration guide
- [config.yaml](config.yaml) - YAML configuration

---

**Last Updated:** 2026-04-14
**Version:** 1.0.0

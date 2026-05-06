# 📋 Pipeline Files Mapping
## ไฟล์ที่ใช้ในแต่ละ Phase สำหรับสร้าง Standalone Notebook

---

## 🔧 **SETUP (ทุก Phase ต้องใช้)**

### ไฟล์: `src/config.py`
**ใช้สำหรับ:** โหลด configuration
**Code ที่ต้อง copy:**
```python
# Lines 16-119: Dataclass definitions
@dataclass
class ProjectConfig: ...
@dataclass
class DatasetConfig: ...
@dataclass
class ModelConfig: ...
@dataclass
class RotationConfig: ...
@dataclass
class MLflowConfig: ...
@dataclass
class PathConfig: ...

# Lines 148-201: get_config() function
def get_config(yaml_path=None): ...
```

**Dependencies:**
- `from dataclasses import dataclass`
- `from typing import Dict, Optional`
- `from pathlib import Path`
- `import yaml`

---

## 📦 **PHASE 1: YOLO-World Auto-Labeling**

### ไฟล์: `src/yolo_world_labeler.py` (241 lines)

**Code sections ที่ต้อง copy:**

#### 1. Class Definition (Lines 14-26)
```python
class YOLOWorldLabeler:
    def __init__(self, model_name="yolov8x-worldv2.pt"):
        self.model_name = model_name
        self.model = None
```

#### 2. Load Model (Lines 27-43)
```python
def load_model(self, classes: List[str]):
    self.model = YOLOWorld(self.model_name)
    self.model.set_classes(classes)
    return True
```

#### 3. Detect Drums (Lines 45-78)
```python
def detect_drums(self, image_path, conf_threshold=0.3):
    results = self.model.predict(image_path, conf=conf_threshold)
    # Extract detections
    return detections
```

#### 4. Convert to YOLO Format (Lines 80-113)
```python
def convert_to_yolo_format(self, detections, img_width, img_height, class_id=1):
    # Convert xyxy to normalized xywh
    return yolo_labels
```

#### 5. Auto-label Dataset (Lines 115-241)
```python
def auto_label_dataset(self, dataset_path, output_path, target_class, 
                       conf_threshold=0.3, max_images=None):
    # Process images and create labels
    return output_path
```

**Dependencies:**
- `import cv2`
- `import numpy as np`
- `from pathlib import Path`
- `from typing import List, Tuple, Optional`
- `from ultralytics import YOLOWorld`
- `import shutil`

**Usage in Pipeline:**
```python
labeler = YOLOWorldLabeler()
auto_labeled_path = labeler.auto_label_dataset(
    dataset_path='./data/Mixer.yolo26',
    output_path='./auto_labeled',
    target_class='concrete mixer drum',
    conf_threshold=0.3,
    max_images=100
)
```

---

## 📦 **PHASE 2: Data Fusion**

### ไฟล์: `src/data_loader.py` (189 lines)

**Code sections ที่ต้อง copy:**

#### 1. LocalDataset Class (Lines 15-28)
```python
@dataclass
class LocalDataset:
    location: str
    train_images: int = 0
    train_labels: int = 0
```

#### 2. DataLoader Class Init (Lines 31-36)
```python
class DataLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config
```

#### 3. Extract Dataset (Lines 38-71)
```python
def extract_dataset(self, zip_path: Path, extract_dir: Path):
    # Extract zip file
    return LocalDataset(...)
```

#### 4. Load All Datasets (Lines 73-98)
```python
def load_all_datasets(self):
    # Load dataset 1 and 2
    return dataset1, dataset2
```

#### 5. Merge Datasets (Lines 100-144)
```python
def merge_datasets(self, datasets: list, output_dir: str):
    # Copy files from all datasets
    # Auto split 80/20 if no validation set
    return merged_dir
```

#### 6. Create data.yaml (Lines 146-189)
```python
def create_data_yaml(self, merged_dir, classes, num_classes):
    # Create YOLO data.yaml
    return data_yaml_path
```

**Dependencies:**
- `import zipfile`
- `import shutil`
- `from pathlib import Path`
- `from dataclasses import dataclass`
- `import yaml`

**Usage in Pipeline:**
```python
loader = DataLoader(config['dataset'])
dataset1, dataset2 = loader.load_all_datasets()
auto_labeled_dataset = LocalDataset(auto_labeled_path)

datasets_to_merge = [dataset1, dataset2, auto_labeled_dataset]
merged_dir = loader.merge_datasets(datasets_to_merge, './merged_dataset')
data_yaml_path = loader.create_data_yaml(merged_dir, classes, num_classes)
```

---

## 📦 **PHASE 3: Label Cleaning**

### ไฟล์: `clean_labels.py` (42 lines)

**Code sections ที่ต้อง copy:**

#### 1. Clean Label File (Lines 9-26)
```python
def clean_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:
            # Remove angle (6th column)
            cleaned_line = ' '.join(parts[:5]) + '\n'
            cleaned_lines.append(cleaned_line)
        elif len(parts) == 5:
            cleaned_lines.append(line)
    
    with open(label_path, 'w') as f:
        f.writelines(cleaned_lines)
```

#### 2. Clean Dataset (Lines 28-42)
```python
def clean_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    
    for split in ['train', 'valid']:
        label_dir = dataset_path / split / 'labels'
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob('*.txt'))
        for label_file in label_files:
            clean_label_file(label_file)
```

**Dependencies:**
- `from pathlib import Path`

**Usage in Pipeline:**
```python
clean_dataset('./merged_dataset')
```

---

## 📦 **PHASE 4: Two-Stage Training**

### ไฟล์: `src/advanced_trainer.py` (140 lines)

**Code sections ที่ต้อง copy:**

#### 1. AdvancedTrainer Class (Lines 11-24)
```python
class AdvancedTrainer(ModelTrainer):
    def __init__(self, model_config, mlflow_config, path_config):
        super().__init__(model_config, mlflow_config, path_config)
```

#### 2. Freeze Layers (Lines 26-47)
```python
def freeze_backbone(self, num_layers=10):
    # Freeze first N layers
    for i, (name, param) in enumerate(self.model.model.named_parameters()):
        if i < num_layers:
            param.requires_grad = False
```

#### 3. Unfreeze Layers (Lines 49-60)
```python
def unfreeze_all(self):
    for param in self.model.model.parameters():
        param.requires_grad = True
```

#### 4. Print Model Info (Lines 62-91)
```python
def print_model_info(self):
    # Count trainable/frozen parameters
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Frozen: {frozen_params:,}")
```

#### 5. Train with Freezing (Lines 93-140)
```python
def train_with_freezing(self, data_yaml_path, freeze_epochs=20):
    # Stage 1: Freeze backbone
    self.freeze_backbone(num_layers=10)
    best_model_path, metrics = self.train(data_yaml_path)
    
    # Stage 2: Fine-tune all
    self.unfreeze_all()
    best_model_path, final_metrics = self.train(data_yaml_path)
    
    return best_model_path, final_metrics
```

**Dependencies:**
- `from trainer import ModelTrainer`
- `from ultralytics import YOLO`

**Usage in Pipeline:**
```python
trainer = AdvancedTrainer(config['model'], config['mlflow'], config['paths'])
trainer.setup_mlflow()
trainer.load_model()
trainer.print_model_info()

best_model_path, final_metrics = trainer.train_with_freezing(
    data_yaml_path=str(data_yaml_path),
    freeze_epochs=20
)
```

### ไฟล์: `src/trainer.py` (ต้องใช้เป็น base class)

**Code sections ที่ต้อง copy:**

#### 1. ModelTrainer Class (Lines 13-127)
```python
class ModelTrainer:
    def __init__(self, model_config, mlflow_config, path_config):
        self.model_config = model_config
        self.mlflow_config = mlflow_config
        self.path_config = path_config
        self.model = None
    
    def setup_mlflow(self):
        # Setup MLflow tracking
        
    def load_model(self):
        # Load YOLO model
        
    def train(self, data_yaml_path):
        # Train model
        return best_model_path, metrics
    
    def validate(self, model_path, data_yaml_path):
        # Validate model
        return results
```

**Dependencies:**
- `from ultralytics import YOLO`
- `import mlflow`
- `import os`
- `from pathlib import Path`

---

## 📦 **PHASE 5: Validation**

**ใช้ไฟล์เดียวกับ Phase 4:** `src/trainer.py`

**Code section:**
```python
def validate(self, model_path, data_yaml_path):
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    return results
```

**Usage in Pipeline:**
```python
val_results = trainer.validate(best_model_path, str(data_yaml_path))
```

---

## 📦 **PHASE 6: ONNX Export & Web Deployment**

### ไฟล์: `src/web_deployer.py` (344 lines)

**Code sections ที่ต้อง copy:**

#### 1. WebDeployer Class (Lines 13-28)
```python
class WebDeployer:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
```

#### 2. Export ONNX (Lines 30-76)
```python
def export_onnx(self, output_dir="./onnx_models", imgsz=640):
    model = YOLO(self.model_path)
    onnx_path = model.export(format='onnx', imgsz=imgsz)
    return onnx_path
```

#### 3. Create Web Package (Lines 78-344)
```python
def create_web_package(self, onnx_path, class_names, output_dir="./web_deployment"):
    # Create HTML, JS, CSS files
    # Copy ONNX model
    # Generate README
    return output_dir
```

**Dependencies:**
- `from ultralytics import YOLO`
- `from pathlib import Path`
- `import shutil`
- `import json`

**Usage in Pipeline:**
```python
deployer = WebDeployer(best_model_path)
onnx_path = deployer.export_onnx()
web_dir = deployer.create_web_package(
    onnx_path=str(onnx_path),
    class_names=config['dataset'].CLASSES
)
```

---

## 📊 **สรุปไฟล์ทั้งหมดที่ต้อง Copy**

| Phase | ไฟล์หลัก | ไฟล์เสริม | จำนวนบรรทัด |
|-------|---------|-----------|-------------|
| **Setup** | `config.py` | - | ~200 |
| **Phase 1** | `yolo_world_labeler.py` | - | 241 |
| **Phase 2** | `data_loader.py` | - | 189 |
| **Phase 3** | `clean_labels.py` | - | 42 |
| **Phase 4** | `advanced_trainer.py` | `trainer.py` | 140 + 127 |
| **Phase 5** | (ใช้ `trainer.py`) | - | - |
| **Phase 6** | `web_deployer.py` | - | 344 |

**รวมทั้งหมด:** ~1,283 lines of code

---

## 🎯 **วิธีสร้าง Standalone Notebook**

### Cell 1: Imports & Config
```python
# Copy from config.py (lines 1-201)
```

### Cell 2: YOLO-World Labeler
```python
# Copy from yolo_world_labeler.py (lines 1-241)
```

### Cell 3: Data Loader
```python
# Copy from data_loader.py (lines 1-189)
```

### Cell 4: Label Cleaner
```python
# Copy from clean_labels.py (lines 1-42)
```

### Cell 5: Trainer (Base)
```python
# Copy from trainer.py (lines 1-127)
```

### Cell 6: Advanced Trainer
```python
# Copy from advanced_trainer.py (lines 1-140)
```

### Cell 7: Web Deployer
```python
# Copy from web_deployer.py (lines 1-344)
```

### Cell 8: Run Pipeline
```python
# Copy from run_complete_pipeline.py (lines 19-140)
```

---

## ⚠️ **สิ่งที่ต้องระวัง**

1. **Import Order:** ต้อง import ตามลำดับ (config → labeler → loader → trainer → deployer)
2. **Dependencies:** ต้องติดตั้ง packages ก่อน:
   ```bash
   pip install ultralytics opencv-python pyyaml mlflow
   ```
3. **File Paths:** ใช้ relative paths ทั้งหมด (`./data`, `./merged_dataset`)
4. **Config:** สามารถใช้ hardcode config แทน YAML ได้

---

## 📝 **Template Notebook Structure**

```
# Cell 1: Install Dependencies
!pip install ultralytics opencv-python pyyaml mlflow

# Cell 2-8: Copy code from files above

# Cell 9: Run Complete Pipeline
config = get_config()
# ... run all phases ...

# Cell 10: Visualize Results
# Show metrics, plots, etc.
```

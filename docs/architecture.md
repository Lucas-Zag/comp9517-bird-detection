# Bird Detection — Software Architecture

## 1. Project Goal
Build a modular bird detection system using
YOLO and RT-DETR.

## 2. Main Components

### Data Module
- Dataset loading
- Annotation validation
- Preprocessing

### Model Module
- YOLO adapter
- RT-DETR adapter
- Unified prediction interface

### Evaluation Module
- Precision and Recall
- mAP@0.5
- Inference latency
- Error analysis

### Configuration
- Model settings
- Training parameters
- Dataset paths

### Scripts
- Train models
- Run inference
- Evaluate results

## 3. Design Principles
- Separation of Concerns
- Reusability
- Reproducibility
- Testability

## 4. Technology Stack
- Python
- PyTorch
- Ultralytics
- Conda
- Git and GitHub

# COMP9517 — Beach Bird Detection

**UNSW Sydney · COMP9517 Computer Vision · Term 3, 2026**

This repository contains a reproducible computer vision workflow for **single-class bird detection** in coastal imagery using **YOLO11n** and **RT-DETR-L**. It includes dataset exploration, model fine-tuning, validation failure analysis, and evaluation utilities. The objective is to detect and localize birds with bounding boxes, including small targets and dense flocks.

> **Scope:** This repository documents the YOLO/RT-DETR experiments. The wider COMP9517 group project evaluates additional computer vision methods separately.

## 1. Dataset

**Beach Bird Detection Dataset v1.0** (Kai Liu, Kaggle, September 2026): [Dataset page](https://www.kaggle.com/datasets/liugalhigh/beach-bird)

| Split | Images | Usage |
| --- | ---: | --- |
| Train | 1,369 | Model training |
| Validation | 291 | Model selection and error analysis |
| Test | 240 | Final held-out evaluation |

Annotations use **YOLO bounding-box format**, with one class: `0 = bird`. The images include negative scenes, sparse groups, and dense flocks. Test images must not be used for hyperparameter selection.

Download the dataset separately and organise it as follows:

```text
data/
├── beach_bird.yaml
└── beach_bird/
    └── Beach Bird Detection Dataset v1.0/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
```

The dataset YAML should point to the extracted directory. For example:

```yaml
path: /absolute/path/to/comp9517-bird-detection/data/beach_bird/Beach Bird Detection Dataset v1.0
train: images/train
val: images/val
test: images/test
names:
  0: bird
```

**Important:** Update `path:` for your own computer. The original images and annotations are not distributed in this repository.

## 2. Project layout

The following paths describe the project's experiment organisation. Some generated folders are local-only and may not be committed to Git.

```text
comp9517-bird-detection/
├── configs/                    # Training configurations
├── data/
│   └── beach_bird.yaml         # Ultralytics dataset configuration
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_yolo_training.ipynb
│   ├── 03_validation_failure_analysis.ipynb
│   └── 04_rtdetr_training.ipynb
├── scripts/                    # Reusable utilities and smoke tests
├── src/
│   └── bird_detection/
│       ├── models/             # Detector interfaces / model adapters
│       └── evaluation/         # IoU and detection error analysis
├── runs/                       # Generated checkpoints / experiment logs (local)
└── README.md
```

Notebook names and run folders may differ between experiment revisions. Refer to the local `configs/` files and each notebook for the exact settings used in a particular run.

## 3. Environment setup

The experiments were developed on **Apple M4 Pro (Apple Silicon)** with Python **3.11**, PyTorch, Ultralytics and the **MPS** backend. CUDA or CPU can also be used where supported; update the `device` setting accordingly.

```bash
git clone https://github.com/Lucas-Zag/comp9517-bird-detection.git
cd comp9517-bird-detection

conda create -n comp9517 python=3.11 -y
conda activate comp9517

python -m pip install ultralytics torch torchvision numpy pandas matplotlib pyyaml jupyterlab
```

If this checkout includes a `pyproject.toml`, install the local `src/` package as well:

```bash
python -m pip install -e .
```

Verify the backend:

```python
import torch
print("MPS available:", torch.backends.mps.is_available())
```

Start the notebooks with `jupyter lab` from the repository root. When working inside `notebooks/`, ensure the notebook resolves the root directory before accessing `data/`, `configs/`, or `runs/`.

## 4. Models and training

| Setting | YOLO11n | RT-DETR-L |
| --- | --- | --- |
| Pretrained checkpoint | `yolo11n.pt` | `rtdetr-l.pt` |
| Image size | 640 × 640 | 640 × 640 |
| Training batch size | 4 | 2 |
| Device | MPS | MPS |
| Main experimental workflow | `02_yolo_training.ipynb` | `04_rtdetr_training.ipynb` |

- **YOLO11n:** lightweight convolutional object detector used for baseline detection and failure analysis. A separate follow-up experiment investigated possible improvements.
- **RT-DETR-L:** real-time detection Transformer with an HGNetv2 backbone, efficient hybrid encoder, and Transformer decoder. The stable fine-tuning experiment used **35 epochs**, **AdamW**, an initial learning rate of **0.0005**, cosine scheduling, and **AMP disabled**.

Use the corresponding notebooks and `configs/` files to reproduce training. Checkpoint names in `runs/train/` are experiment-specific; Ultralytics may append a suffix when an output directory already exists.

### Example inference

Set `weights` to the checkpoint saved by the relevant training run:

```python
from ultralytics import RTDETR

weights = "runs/train/<your-rtdetr-run>/weights/best.pt"
model = RTDETR(weights)

results = model.predict(
    source="path/to/image.jpg",
    imgsz=640,
    conf=0.25,
    device="mps",  # change to "cpu" or "0" if appropriate
    save=True,
)
```

For YOLO11n, import `YOLO` from `ultralytics` and load the corresponding YOLO checkpoint.

## 5. Evaluation protocol

Final evaluation uses the **240-image held-out test set**. In fixed-threshold error analysis:

- **Confidence threshold:** `0.25` (retain predictions with confidence ≥ 0.25).
- **Matching IoU threshold:** `0.50` (one-to-one predicted/ground-truth box matching).
- **Metrics:** TP, FP, FN, precision, recall, F1, and counting MAE where available.
- **AP metrics:** `mAP@0.5` and `mAP@0.5:0.95` are reported separately using the standard confidence-ranked evaluation procedure. Their IoU thresholds must not be confused with the confidence cutoff.

Precision, recall and F1 at one fixed confidence threshold are **not** interchangeable with standard mAP scores. If comparing another method, use the same test images and matching protocol.

## 6. Results

Selected final held-out test results from the recorded experiments:

| Model | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ |
| --- | ---: | ---: |
| YOLO11n | 0.8712 | 0.4835 |
| RT-DETR-L | **0.9119** | **0.5264** |

Additional fixed-threshold results at **confidence = 0.25, matching IoU = 0.50**:

| Model | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 1,276 | 411 | 242 | 0.7564 | 0.8406 | 0.7963 |
| RT-DETR-L | 1,412 | 954 | 106 | 0.5968 | 0.9302 | 0.7271 |

These results illustrate a **precision–recall trade-off**: RT-DETR-L retrieves more annotated birds at the selected confidence threshold, while producing more unmatched detections. Standard mAP evaluates performance across confidence levels and should be interpreted separately.

The reported inference times depend strongly on hardware, batch size, backend and preprocessing scope. Cross-device latency is therefore **not treated as a controlled algorithm ranking**.

## 7. Reproducibility notes

1. Use the provided dataset split without moving images between train, validation and test.
2. Record the model checkpoint, random seed, image size, batch size, device and software versions for each experiment.
3. Keep each experiment's `args.yaml`, `results.csv`, plots and `best.pt` together locally.
4. Use validation results for model selection; evaluate the held-out test set only after finalising the configuration.
5. Do not commit large datasets, trained weights or generated result images to the source-code submission.

## 8. References and acknowledgements

- Kai Liu, *Beach Bird Detection Dataset v1.0*, Kaggle, 2026. <https://www.kaggle.com/datasets/liugalhigh/beach-bird>
- Y. Zhao et al., “DETRs Beat YOLOs on Real-time Object Detection,” *CVPR*, 2024. <https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html>
- Ultralytics, *Ultralytics YOLO and RT-DETR documentation*. <https://docs.ultralytics.com/>

This code was developed for the **UNSW COMP9517 Computer Vision group project**. Upstream software and pretrained weights remain subject to their respective licences. No new licence is asserted for this repository by this README.

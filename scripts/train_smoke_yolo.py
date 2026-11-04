from pathlib import Path
import json
import time

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# Load pretrained model
model = YOLO("yolo11n.pt")

# Start timing
start = time.perf_counter()

# Run a small training experiment
model.train(
    data=str(ROOT / "data/beach_bird.yaml"),
    epochs=1,
    fraction=0.05,
    imgsz=640,
    batch=4,
    device="mps",
    workers=0,
    seed=42,
    plots=True,
    project=str(ROOT / "runs/train"),
    name="yolo11n_smoke",
    exist_ok=False,
)

elapsed = time.perf_counter() - start
run_dir = Path(model.trainer.save_dir)

# Save experiment timing
(run_dir / "run_summary.json").write_text(
    json.dumps({
        "wall_time_seconds": round(elapsed, 2),
        "model": "yolo11n.pt",
        "device": "mps",
        "experiment": "smoke_test"
    }, indent=2)
)

print(f"Training time: {elapsed:.2f} seconds")
print(f"Results directory: {run_dir}")

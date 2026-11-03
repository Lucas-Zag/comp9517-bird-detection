from pathlib import Path
from ultralytics import YOLO

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Load pretrained model
model = YOLO("yolo11n.pt")

# Run inference
results = model.predict(
    source=str(ROOT / "data" / "bus.jpg"),
    device="mps",
    save=True,
    project=str(ROOT / "runs" / "predict"),
    name="python_yolo",
)

# Print detection count
print(f"Detected objects: {len(results[0].boxes)}")

# Inspect individual detections
result = results[0]

for i, box in enumerate(result.boxes):
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].tolist()

    print(
        f"Object {i + 1}: "
        f"class={result.names[class_id]}, "
        f"confidence={confidence:.2f}, "
        f"box={coordinates}"
    )

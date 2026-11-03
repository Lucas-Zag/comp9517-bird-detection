from .base import Detector
from .yolo import YOLODetector
from .rtdetr import RTDETRDetector


def create_detector(
    model_name: str,
    model_path: str | None = None,
    device: str = "mps",
) -> Detector:
    """Create a detector using the selected architecture."""

    name = model_name.lower()

    if name == "yolo":
        return YOLODetector(
            model_path=model_path or "yolo11n.pt",
            device=device,
        )

    if name == "rtdetr":
        return RTDETRDetector(
            model_path=model_path or "rtdetr-l.pt",
            device=device,
        )

    raise ValueError(f"Unsupported detector: {model_name}")

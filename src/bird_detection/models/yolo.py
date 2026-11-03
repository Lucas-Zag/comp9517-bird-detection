from ultralytics import YOLO

from .base import Detection, Detector


class YOLODetector(Detector):
    """YOLO object detector adapter."""

    def __init__(self, model_path="yolo11n.pt", device="mps"):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, image_path: str) -> list[Detection]:
        results = self.model.predict(
            source=image_path,
            device=self.device,
            save=False,
            verbose=False,
        )

        result = results[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            detection = Detection(
                class_id=class_id,
                label=result.names[class_id],
                confidence=float(box.conf.item()),
                xyxy=tuple(box.xyxy[0].tolist()),
            )

            detections.append(detection)

        return detections

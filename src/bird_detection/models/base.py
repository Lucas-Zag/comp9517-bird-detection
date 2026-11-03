from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Detection:
    """Standard output for one detected object."""

    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def predict(self, image_path: str) -> list[Detection]:
        """Detect objects and return standardised results."""
        pass

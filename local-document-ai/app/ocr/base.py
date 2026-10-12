"""OCR engine interface.

Every recognition backend (PaddleOCR today, a Transformer OCR tomorrow)
implements ``OCREngine`` and returns ``OCRResult`` items, so the rest of the
pipeline never depends on a specific engine.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from PIL import Image


@dataclass
class OCRResult:
    page: int
    text: str
    bbox: List[float]        # [x1, y1, x2, y2] in pixel coordinates
    confidence: float        # 0..1


class OCREngine(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> List[OCRResult]:
        """Detect and recognize text in one page image (``page`` is set to 0)."""

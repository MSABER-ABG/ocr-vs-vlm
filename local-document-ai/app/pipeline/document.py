"""The document pipeline: load -> preprocess -> OCR -> reading order."""
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

from app.config import Config
from app.io import load_document
from app.layout import Line, sort_reading_order
from app.ocr import OCREngine, create_engine
from app.preprocessing import preprocess


@dataclass
class PageResult:
    page: int
    width: int
    height: int
    lines: List[Line] = field(default_factory=list)


@dataclass
class DocumentResult:
    source: str
    pages: List[PageResult]


class DocumentAI:
    def __init__(self, config: Optional[Config] = None, ocr: Optional[OCREngine] = None):
        self.config = config or Config()
        self.ocr = ocr or create_engine(self.config)

    def process(self, path) -> DocumentResult:
        """Process a PDF/image file from disk."""
        return self.process_pages(load_document(path, self.config), source=str(path))

    def process_pages(self, pages: List[Image.Image], source: str = "memory") -> DocumentResult:
        """Process already-loaded PIL pages (used by the web UI)."""
        results: List[PageResult] = []

        for number, page in enumerate(pages, start=1):
            image, scale = preprocess(page, self.config)
            detections = self.ocr.predict(image)

            # Map boxes back to the ORIGINAL page pixels and tag the page number.
            for d in detections:
                d.page = number
                d.bbox = [round(v / scale, 1) for v in d.bbox]

            lines = sort_reading_order(detections, self.config.line_overlap_ratio)
            results.append(
                PageResult(page=number, width=page.width, height=page.height, lines=lines)
            )
        return DocumentResult(source=source, pages=results)

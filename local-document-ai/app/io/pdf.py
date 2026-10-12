from pathlib import Path
from typing import List

from PIL import Image


def load_pdf(path: Path, dpi: int = 300, max_pages: int = 50) -> List[Image.Image]:
    """Render PDF pages to PIL images using pypdfium2 (no poppler needed)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        scale = dpi / 72.0
        pages = []
        for i in range(min(len(pdf), max_pages)):
            bitmap = pdf[i].render(scale=scale)
            pages.append(bitmap.to_pil().convert("RGB"))
        return pages
    finally:
        pdf.close()

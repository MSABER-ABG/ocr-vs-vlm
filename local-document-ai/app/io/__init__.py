"""Document loading: PDF and image files -> list of PIL pages."""
from pathlib import Path
from typing import List

from PIL import Image

from app.config import Config
from app.io.image import load_image
from app.io.pdf import load_pdf

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def load_document(path, config: Config) -> List[Image.Image]:
    """Return the pages of a PDF or image file as RGB PIL images."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p, dpi=config.pdf_dpi, max_pages=config.max_pages)
    if ext in IMAGE_EXTS:
        return load_image(p)
    raise ValueError(f"Unsupported file type: {ext}")

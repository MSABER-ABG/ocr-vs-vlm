from pathlib import Path
from typing import List

from PIL import Image, ImageOps


def load_image(path: Path) -> List[Image.Image]:
    """Load an image file (multi-frame TIFF supported), honoring EXIF rotation."""
    img = Image.open(path)
    pages = []
    n_frames = getattr(img, "n_frames", 1)
    for i in range(n_frames):
        img.seek(i)
        frame = ImageOps.exif_transpose(img.copy()).convert("RGB")
        pages.append(frame)
    return pages

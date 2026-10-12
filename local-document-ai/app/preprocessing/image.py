from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.config import Config


def preprocess(page: Image.Image, config: Config) -> Tuple[Image.Image, float]:
    """Prepare a page for OCR.

    Returns ``(image, scale)`` where ``scale`` is the factor applied to the
    original page. The pipeline divides OCR coordinates by ``scale`` so all
    bounding boxes in the output refer to the *original* page pixels.

    Steps: upscale small pages, then (optionally) remove the slowly-varying
    background. Cheques, certificates and ID cards carry guilloche/watermark
    patterns that confuse text detection; dividing the image by a heavily
    blurred copy of itself flattens the paper and leaves only the ink.
    """
    img = page.convert("RGB")
    if not config.preprocess:
        return img, 1.0

    scale = 1.0
    long_side = max(img.size)
    if config.max_upscale > 1.0 and long_side < config.upscale_target_long:
        scale = min(config.max_upscale, config.upscale_target_long / long_side)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if config.normalize_background:
        g = gray.astype(np.float32)
        sigma = max(15.0, 0.012 * max(g.shape))
        bg = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
        gray = np.clip(g / (bg + 1e-3) * 235.0, 0, 255).astype(np.uint8)
    else:
        lo, hi = np.percentile(gray, (1, 99))
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    out = Image.fromarray(gray)
    if config.sharpen:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return out.convert("RGB"), scale

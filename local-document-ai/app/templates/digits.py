"""Glyph-by-glyph reading of a printed digit string (e.g. the 14-digit national ID).

Why not just run OCR on the band? In the card's font the Arabic-Indic zero is a
tiny dot, and text recognizers silently drop it. So we:
  1. binarize the band and find connected components (glyphs),
  2. treat small round components as the digit zero,
  3. recognize every tall glyph on its own with the recognizer (rec-only).
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

_AR_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EXT_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_TO_WESTERN = {**{c: str(i) for i, c in enumerate(_AR_INDIC)},
               **{c: str(i) for i, c in enumerate(_EXT_INDIC)},
               **{str(i): str(i) for i in range(10)}}


def to_western_digits(text: str) -> str:
    """Keep only digits, converting Arabic-Indic / Persian digits to 0-9."""
    return "".join(_TO_WESTERN[c] for c in text if c in _TO_WESTERN)


@dataclass
class Glyph:
    x: int
    y: int
    w: int
    h: int
    is_dot: bool
    text: str = ""
    confidence: float = 1.0


def _components(gray: np.ndarray, c: int, min_area: int):
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, c
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [
        (int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]))
        for s in stats[1:]
        if s[4] >= min_area
    ]


def _overlaps(a, b) -> bool:
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _h_gap(a, b) -> float:
    """Horizontal distance between two (x, width) intervals (0 if they touch/overlap)."""
    return max(0.0, max(a[0], b[0]) - min(a[0] + a[1], b[0] + b[1]))


def segment_glyphs(band: Image.Image) -> List[Glyph]:
    """Find digit glyphs and zero-dots in a band that contains ONE line of digits."""
    gray = cv2.cvtColor(np.array(band.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    edge = max(6, int(0.015 * W))                        # ignore border noise

    # Dark ink -> tall digit glyphs (and the zero dots, handled below).
    strict = [c for c in _components(gray, 18, 25)
              if c[0] > edge and c[0] + c[2] < W - edge]
    if not strict:
        return []
    heights = sorted(c[3] for c in strict)
    tall_ref = heights[int(len(heights) * 0.75)] if len(heights) > 3 else heights[-1]
    tall = [c for c in strict if c[3] >= 0.6 * tall_ref]
    line_cy = float(np.median([c[1] + c[3] / 2 for c in tall]))
    h_med = float(np.median([c[3] for c in tall]))

    glyph_boxes = [c for c in tall if abs((c[1] + c[3] / 2) - line_cy) <= 0.9 * h_med]
    glyphs = [Glyph(x, y, w, h, is_dot=False) for x, y, w, h, _ in glyph_boxes]

    # The zero is a small solid dot/diamond. Accept compact, filled, roughly round
    # blobs on the digit line that are clearly smaller than a digit and do not
    # overlap one (broken pieces of digits overlap their glyph box).
    for x, y, w, h, area in strict:
        cy = y + h / 2
        if abs(cy - line_cy) > 0.7 * h_med:
            continue
        if not (0.2 * h_med <= h <= 0.55 * h_med and 0.2 * h_med <= w <= 0.55 * h_med):
            continue
        if not (0.6 <= w / max(h, 1) <= 1.7) or area / float(w * h) < 0.5:
            continue
        if any(_overlaps((x, y, w, h), gb) for gb in glyph_boxes):
            continue
        # A real zero sits in a gap between digits; a fragment of a blurry digit
        # touches its glyph. Require clear horizontal space to every digit.
        if any(_h_gap((x, w), (gb[0], gb[2])) < 0.2 * h_med for gb in glyph_boxes):
            continue
        glyphs.append(Glyph(x, y, w, h, is_dot=True, text="0"))

    glyphs.sort(key=lambda g: g.x)
    return glyphs


def _pad_square(img: Image.Image, pad: int = 14) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (235, 235, 235))
    canvas.paste(img, (pad, pad))
    return canvas


def _tile(crop: Image.Image, copies: int = 3) -> Image.Image:
    """Repeat one glyph side by side so the line recognizer sees text-like input.

    A text recognizer expects a line with context and often returns nothing for a
    lone thin glyph such as 1 or 7; three spaced copies are read reliably."""
    gap = max(8, int(0.5 * crop.width))
    w, h = crop.size
    pad = 14
    canvas = Image.new("RGB", (copies * w + (copies - 1) * gap + 2 * pad, h + 2 * pad), (235, 235, 235))
    for i in range(copies):
        canvas.paste(crop, (pad + i * (w + gap), pad))
    return canvas


def _recognize_glyph(crop: Image.Image, reader):
    """Majority vote over the characters the recognizer returns for the tiled glyph."""
    out = reader.ocr(np.array(_tile(crop)), det=False, cls=False)
    if not out or not out[0]:
        return "", 0.0
    text, conf = out[0][0]
    digits = to_western_digits(text)
    if not digits:
        return "", float(conf)
    best = max(set(digits), key=digits.count)           # most frequent digit
    return best, float(conf)


def read_digit_line(band: Image.Image, reader, min_conf: float = 0.0):
    """Return ``(digits, glyphs)`` for a band holding one printed digit line.

    ``reader`` is a PaddleOCR instance (Arabic model). Recognition of each tall
    glyph is done with ``det=False`` so no text detection is involved.
    """
    glyphs = segment_glyphs(band)
    arr = np.array(band.convert("RGB"))
    for g in glyphs:
        if g.is_dot:
            continue
        m = max(2, int(0.12 * g.h))
        crop = Image.fromarray(
            arr[max(0, g.y - m): g.y + g.h + m, max(0, g.x - m): g.x + g.w + m]
        )
        g.text, g.confidence = _recognize_glyph(crop, reader)
    digits = "".join(g.text for g in glyphs)
    return digits, glyphs

"""Fast national-number reader for the Egyptian ID card (no OCR model, ~0.1 s).

Adapted from ``id_card_reader.py`` written by a colleague: the digit line is cut into glyphs and every glyph
is classified by a tiny softmax model (the weights below are theirs, stored zlib+base64 as float16). The card's
printed "0" looks like a dot, so a dot-shaped glyph is read as 0. The result is validated with the same
``validate_national_id`` the rest of the engine uses.

Measured on the sample cards of this project it reads the same number as ABG OCR where ABG reads one,
and it needs no PaddlePaddle, so it can be used next to PaddleOCR-VL (which misreads this digit font).
"""
import base64
import zlib

import cv2
import numpy as np
from PIL import Image

from .egypt_id import validate_national_id

CARD_W, CARD_H = 1600, 1009            # the size the zones below were tuned for
DIGITS_EXPECTED = 14
MIN_DIGIT_CONFIDENCE = 0.80
# Where the number line can be, as fractions of the straightened card (x1, y1, x2, y2).
NUMBER_ZONES = [
    (0.35, 0.70, 1.00, 0.92),
    (0.35, 0.66, 1.00, 0.88),
    (0.35, 0.74, 1.00, 0.96),
    (0.25, 0.70, 1.00, 0.92),
    (0.35, 0.62, 1.00, 0.84),
]
THRESHOLD_C_VALUES = (18, 12, 25)


# -------------------------- glyph segmentation ---------------------------------

def segment_glyphs(band, c_value):
    """Split ONE printed digit line into glyphs: [{x, y, w, h, is_dot}] left to right."""
    gray = cv2.GaussianBlur(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    H, W = gray.shape
    block = max(31, int(H * 0.23) // 2 * 2 + 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c_value)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    edge, min_area = max(6, int(0.01 * W)), max(30, int(0.0012 * H * H))
    comps = [dict(x=int(s[0]), y=int(s[1]), w=int(s[2]), h=int(s[3])) for s in stats[1:]
             if s[4] >= min_area and s[0] > edge and s[0] + s[2] < W - edge]
    if len(comps) < 3:
        return []

    heights = sorted(c["h"] for c in comps)
    tall = [c for c in comps if c["h"] >= 0.6 * heights[int(len(heights) * 0.75)]]
    if len(tall) < 3:
        return []
    line_cy = float(np.median([c["y"] + c["h"] / 2 for c in tall]))
    h_med = float(np.median([c["h"] for c in tall]))
    comps = [c for c in comps if abs(c["y"] + c["h"] / 2 - line_cy) <= 0.9 * h_med]

  
    comps.sort(key=lambda c: c["x"])
    merged = []
    for c in comps:
        if merged:
            m = merged[-1]
            new_w = max(m["x"] + m["w"], c["x"] + c["w"]) - m["x"]
            if c["x"] - (m["x"] + m["w"]) < 0.14 * h_med and new_w <= 1.2 * h_med:
                x1, y1 = min(m["x"], c["x"]), min(m["y"], c["y"])
                merged[-1] = dict(x=x1, y=y1, w=new_w, h=max(m["y"] + m["h"], c["y"] + c["h"]) - y1)
                continue
        merged.append(dict(c))

    glyphs = [g for g in merged if g["h"] >= 0.3 * h_med]
    for g in glyphs:
        g["is_dot"] = g["h"] <= 0.55 * h_med and g["w"] <= 0.55 * h_med and 0.5 <= g["w"] / max(g["h"], 1) <= 2.0
    return glyphs


def find_number_line(card):
    """Try the candidate zones; keep the first that splits into exactly 14 glyphs (else the closest)."""
    best = None
    for zone in NUMBER_ZONES:
        x1, y1, x2, y2 = zone
        ox, oy = int(x1 * CARD_W), int(y1 * CARD_H)
        band = card[oy:int(y2 * CARD_H), ox:int(x2 * CARD_W)]
        for c_value in THRESHOLD_C_VALUES:
            glyphs = segment_glyphs(band, c_value)
            key = -abs(len(glyphs) - DIGITS_EXPECTED)
            if best is None or key > best["key"]:
                best = dict(key=key, offset=(ox, oy), glyphs=glyphs)
            if key == 0:
                return best
    return best


def glyph_masks(card, line):
    """Ink masks (255 = ink), one per glyph, from a shadow-flattened and Otsu-thresholded band."""
    ox, oy = line["offset"]
    gl = line["glyphs"]
    x1 = max(0, ox + min(g["x"] for g in gl) - 20)
    x2 = ox + max(g["x"] + g["w"] for g in gl) + 20
    y1 = max(0, oy + min(g["y"] for g in gl) - 20)
    y2 = oy + max(g["y"] + g["h"] for g in gl) + 20
    gray = cv2.cvtColor(card[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    k = int(gray.shape[0] * 0.8) // 2 * 2 + 1
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (k, k), 0)
    flat = np.clip(gray.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)
    bw = cv2.threshold(cv2.GaussianBlur(flat, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    masks = []
    for g in gl:
        gx, gy = ox + g["x"] - x1, oy + g["y"] - y1
        masks.append(bw[max(0, gy - 3): gy + g["h"] + 3, max(0, gx - 3): gx + g["w"] + 3])
    return masks


# ------------------------------ digit classifier --------------------------------

_S = 24
_weights = None

def glyph_feature(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    th = _S - 4
    tw = max(2, min(_S - 2, int(round(w * th / h))))
    crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_S, _S), np.uint8)
    y0, x0 = (_S - th) // 2, (_S - tw) // 2
    canvas[y0:y0 + th, x0:x0 + tw] = crop
    return (cv2.GaussianBlur(canvas, (3, 3), 0.8).astype(np.float32) / 255.0).ravel()


def classify_digit(mask):
    """-> (digit as str, probability). '?' when the glyph is empty."""
    global _weights
    if _weights is None:
        raw = zlib.decompress(base64.b64decode("".join(_WEIGHTS_B64)))
        n = 9 * _S * _S
        _weights = (np.frombuffer(raw[:2 * n], np.float16).astype(np.float32).reshape(9, _S * _S),
                    np.frombuffer(raw[2 * n:], np.float16).astype(np.float32))
    feature = glyph_feature(mask)
    if feature is None:
        return "?", 0.0
    z = _weights[0] @ feature + _weights[1]
    p = np.exp(z - z.max())
    p /= p.sum()
    return str(int(p.argmax()) + 1), float(p.max())


_WEIGHTS_B64 = (
    "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"
    "T/YGd/jG7f3DXOP5NZLD+JgqzP8Bpd1dWg=="
)


def read_number_fast(card: Image.Image) -> dict:
    """Read the 14-digit number from a STRAIGHTENED card (PIL). Same dict as ``validate_national_id`` plus
    ``confidence`` ("high" / "medium" / "low"), ``digit_probabilities`` and ``source``."""
    bgr = cv2.cvtColor(np.array(card.convert("RGB")), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)
    line = find_number_line(bgr)
    if not line or not line["glyphs"]:
        result = validate_national_id("")
        result.update(confidence="low", digit_probabilities=[], source="glyph classifier")
        return result
    reads = []
    for g, mask in zip(line["glyphs"], glyph_masks(bgr, line)):
        reads.append(("0", 1.0) if g["is_dot"] else classify_digit(mask))
    result = validate_national_id("".join(d for d, _ in reads))
    weakest = min(p for _, p in reads)
    result["confidence"] = "low" if not result["valid"] else ("high" if weakest >= MIN_DIGIT_CONFIDENCE else "medium")
    result["digit_probabilities"] = [round(p, 3) for _, p in reads]
    result["source"] = "glyph classifier"
    return result

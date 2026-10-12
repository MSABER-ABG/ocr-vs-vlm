"""ID-1 card detection and perspective correction (OpenCV, fully local)."""
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ISO/IEC 7810 ID-1 (85.60 x 53.98 mm). Warp to a fixed, generous canvas.
CARD_W, CARD_H = 1712, 1080


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32",
    )


def find_card_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Locate the card as a 4-point quadrilateral (in original pixel coordinates).

    Returns None when no plausible card is found (caller falls back to the whole image).
    """
    h, w = bgr.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    # Cards are brighter than the surface they lie on.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(best) / float(small.shape[0] * small.shape[1])
    if area_ratio < 0.20:
        return None

    # Rounded corners make approxPolyDP unreliable; the min-area rectangle of the
    # convex hull is robust for a mostly-rectangular card.
    hull = cv2.convexHull(best)
    quad = cv2.boxPoints(cv2.minAreaRect(hull))
    quad = _order_points(quad) / scale
    return quad


def warp_card(image: Image.Image, quad: Optional[np.ndarray]) -> Image.Image:
    """Perspective-correct the card to a canonical landscape canvas."""
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if quad is None:
        h, w = bgr.shape[:2]
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    # Make sure the long side maps to the canvas width (landscape).
    tl, tr, br, bl = quad
    if np.linalg.norm(tr - tl) < np.linalg.norm(bl - tl):
        quad = np.array([bl, tl, tr, br], dtype="float32")

    dst = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], dtype="float32")
    m = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    out = cv2.warpPerspective(bgr, m, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def extract_card(image: Image.Image) -> Image.Image:
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return warp_card(image, find_card_quad(bgr))

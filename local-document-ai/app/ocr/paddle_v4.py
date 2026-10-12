"""PaddleOCR recognition engine (CPU or NVIDIA GPU, fully local)."""
import re
from typing import Dict, List

import numpy as np
from PIL import Image

from app.config import Config
from app.ocr.base import OCREngine, OCRResult

_ARABIC = re.compile(r"[؀-ۿݐ-ݿ]")
# Anything that is not Arabic script or whitespace: digits, Latin, punctuation.
_NON_ARABIC_RUN = re.compile(r"[^؀-ۿݐ-ݿ\s]+")
_ARABIC_LANGS = {"ar", "fa", "ug", "ur"}


def to_logical_order(text: str) -> str:
    """Convert Arabic text that came out in VISUAL order into logical order.

    PaddleOCR's Arabic recognizer returns characters left-to-right as drawn,
    i.e. each Arabic word is spelled backwards and multi-word boxes have their
    words in reverse. Reversing the whole string fixes both; digit/Latin runs
    are then flipped back so numbers keep their natural left-to-right order.
    Text without Arabic letters is returned unchanged.
    """
    if not _ARABIC.search(text):
        return text
    return _NON_ARABIC_RUN.sub(lambda m: m.group(0)[::-1], text[::-1])


def _overlap(a: List[float], b: List[float]) -> float:
    """Intersection area / area of the smaller box (robust to word-vs-line splits)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def merge_by_priority(passes: List[List[OCRResult]], threshold: float) -> List[OCRResult]:
    """Combine per-language passes. The first pass wins; later passes only add
    boxes in regions where no earlier pass found text. This avoids duplicates and
    stops a wrong-language model from overriding the right one with garbage."""
    merged: List[OCRResult] = list(passes[0]) if passes else []
    for extra in passes[1:]:
        for r in extra:
            if all(_overlap(r.bbox, m.bbox) < threshold for m in merged):
                merged.append(r)
    return merged


def merge_by_confidence(passes: List[List[OCRResult]], threshold: float) -> List[OCRResult]:
    """Combine per-language passes by picking, for every region, the box the models
    agree is most confident (greedy non-maximum suppression on overlap).

    Each model is best at its own script: the Arabic model reads Arabic text and
    Arabic-Indic digits, the English model reads Latin text and serial numbers
    (e.g. ``AB1234567``). Letting confidence decide per region beats forcing one
    language to win everywhere."""
    allboxes = [r for p in passes for r in p]
    kept: List[OCRResult] = []
    for r in sorted(allboxes, key=lambda x: x.confidence, reverse=True):
        if all(_overlap(r.bbox, k.bbox) < threshold for k in kept):
            kept.append(r)
    return kept


def make_reader(lang: str, config: Config):
    """Create a single PP-OCRv4 PaddleOCR reader (used for digit-by-digit reading)."""
    try:
        import torch  # noqa: F401  (must be imported before paddle on Windows)
    except ImportError:
        pass
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=config.use_gpu, show_log=False)


class PaddleEngine(OCREngine):
    def __init__(self, config: Config):
        # On Windows, torch must be imported before paddle, otherwise torch's
        # shm.dll fails to load (both ship their own libiomp5md.dll).
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        from paddleocr import PaddleOCR

        self.config = config
        # dict preserves insertion order: first language = primary.
        self._readers: Dict[str, "PaddleOCR"] = {
            lang: PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=config.use_gpu,
                show_log=False,
            )
            for lang in config.languages
        }

    def get_reader(self, lang: str):
        """Underlying PaddleOCR instance for ``lang`` (used by templates for digit reading)."""
        return self._readers[lang]

    def _run(self, lang: str, reader, arr) -> List[OCRResult]:
        fix_order = self.config.arabic_logical_order and lang in _ARABIC_LANGS
        out = reader.ocr(arr, cls=True)
        results: List[OCRResult] = []
        for poly, (text, conf) in (out[0] or []):
            if conf < self.config.min_confidence or not text.strip():
                continue
            if fix_order:
                text = to_logical_order(text)
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            results.append(
                OCRResult(
                    page=0,
                    text=text,
                    bbox=[min(xs), min(ys), max(xs), max(ys)],
                    confidence=float(conf),
                )
            )
        return results

    def predict(self, image: Image.Image) -> List[OCRResult]:
        arr = np.array(image.convert("RGB"))
        passes = [self._run(lang, reader, arr) for lang, reader in self._readers.items()]
        if self.config.merge_strategy == "priority":
            return merge_by_priority(passes, self.config.dedupe_overlap)
        return merge_by_confidence(passes, self.config.dedupe_overlap)

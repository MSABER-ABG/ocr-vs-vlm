"""Reading order: group word/box detections into lines, order words per line.

- Lines are ordered top to bottom.
- Inside a line, consecutive boxes of the same script form a *run*:
    * an Arabic run reads right-to-left,
    * a Latin/number run reads left-to-right.
- The runs themselves follow the line's dominant script: right-to-left when most
  boxes are Arabic, otherwise left-to-right. So the English sentence on a
  bilingual cheque stays in order even when an Arabic label shares its row.
"""
import re
from dataclasses import dataclass
from typing import List

from app.ocr.base import OCRResult

# Arabic LETTERS only (digits are neutral: they join whichever run they sit in).
_AR_LETTER = re.compile(r"[ء-يٮ-ۓݐ-ݿ]")


@dataclass
class Line:
    page: int
    text: str
    bbox: List[float]
    confidence: float
    direction: str                 # "rtl" | "ltr"
    words: List[OCRResult]


def _kind(text: str) -> str:
    if _AR_LETTER.search(text):
        return "ar"
    return "other"


def order_line_words(words: List[OCRResult]):
    """Return ``(ordered_words, direction)`` for the boxes of one visual line."""
    by_x = sorted(words, key=lambda w: w.bbox[0])
    kinds = [_kind(w.text) for w in by_x]

    # Digit/punctuation-only boxes are neutral: they join the run before them
    # (or the one after, when they start the line).
    resolved: List[object] = [
        None if not any(c.isalpha() for c in w.text) else k for w, k in zip(by_x, kinds)
    ]
    last = None
    for i, k in enumerate(resolved):
        if k is None:
            resolved[i] = last
        else:
            last = k
    nxt = None
    for i in range(len(resolved) - 1, -1, -1):
        if resolved[i] is None:
            resolved[i] = nxt
        else:
            nxt = resolved[i]
    resolved = [k or "other" for k in resolved]

    runs: List[List[OCRResult]] = []
    run_kind: List[str] = []
    for w, k in zip(by_x, resolved):
        if run_kind and run_kind[-1] == k:
            runs[-1].append(w)
        else:
            runs.append([w])
            run_kind.append(k)

    n_ar = sum(1 for k in resolved if k == "ar")
    rtl_line = n_ar * 2 >= len(resolved) and n_ar > 0

    ordered_runs = []
    for run, k in zip(runs, run_kind):
        ordered_runs.append(sorted(run, key=lambda w: -w.bbox[2]) if k == "ar" else run)
    if rtl_line:
        ordered_runs.reverse()
    return [w for run in ordered_runs for w in run], ("rtl" if rtl_line else "ltr")


def sort_reading_order(results: List[OCRResult], overlap_ratio: float = 0.6) -> List[Line]:
    if not results:
        return []

    heights = [r.bbox[3] - r.bbox[1] for r in results]
    threshold = max(sum(heights) / len(heights) * overlap_ratio, 5.0)

    def cy(r: OCRResult) -> float:
        return (r.bbox[1] + r.bbox[3]) / 2

    groups: List[List[OCRResult]] = []
    for r in sorted(results, key=cy):
        # compare with the running mean y of the current line (tolerates skew)
        if groups and abs(cy(r) - sum(cy(x) for x in groups[-1]) / len(groups[-1])) <= threshold:
            groups[-1].append(r)
        else:
            groups.append([r])

    lines: List[Line] = []
    for g in groups:
        ordered, direction = order_line_words(g)
        lines.append(
            Line(
                page=g[0].page,
                text=" ".join(w.text for w in ordered),
                bbox=[
                    min(w.bbox[0] for w in g),
                    min(w.bbox[1] for w in g),
                    max(w.bbox[2] for w in g),
                    max(w.bbox[3] for w in g),
                ],
                confidence=sum(w.confidence for w in g) / len(g),
                direction=direction,
                words=ordered,
            )
        )
    return lines

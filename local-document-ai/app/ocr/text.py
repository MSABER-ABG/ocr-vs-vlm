"""Text-order helpers shared by the recognition engines."""
import re

# Arabic letters and combining marks (NOT digits, NOT punctuation).
_AR = (
    "ؐ-ؚؠ-ٟٮ-ۓە-ۜ۟-ۨ"
    "۪-ۯۺ-ۿݐ-ݿ"
)
_HAS_ARABIC = re.compile(f"[{_AR}]")
# Everything that is neither Arabic letters nor whitespace: digits (Western,
# Arabic-Indic, Persian), Latin letters, punctuation such as / . : - + %.
_LTR_RUN = re.compile(f"[^{_AR}\\s]+")


def visual_to_logical(text: str) -> str:
    """Convert text in VISUAL (left-to-right glyph) order to logical reading order.

    CTC recognizers read the picture left to right, so an Arabic line comes out
    with its words and the letters of each word reversed. Reversing the whole
    string fixes both; runs of digits / Latin / punctuation are then flipped back
    so numbers and codes keep their natural left-to-right order. Strings with no
    Arabic letters are returned unchanged.
    """
    if not _HAS_ARABIC.search(text):
        return text
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], text[::-1])

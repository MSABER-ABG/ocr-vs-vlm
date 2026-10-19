"""Egyptian ID card through PaddleOCR-VL: straightened card in, validated fields out.

PaddleOCR-VL reads the card's name and address well but misreads the printed digit font (its number often
comes back as Latin look-alikes: ``Y 94.1.1.4 149 04``). So the national number is read by the small glyph
classifier (``id_number_reader``) from the same straightened card, and the VL text is only used for the
name, address and serial. Pure functions: no PaddlePaddle import here.

    card   = extract_card(image)                 # perspective-corrected, fixed size (card.py)
    number = read_number_fast(card)              # id_number_reader.py
    text   = <PaddleOCR-VL text of ``card``>
    fields = build_id_fields(text, number)       # this module
"""
from typing import Any, Dict

from .text_fields import id_fields_from_text


def _has_digits(nid: Dict[str, Any], n: int = 14) -> bool:
    return len(str(nid.get("value", ""))) == n


def pick_number(glyph: Dict[str, Any], from_text: Dict[str, Any]) -> Dict[str, Any]:
    """The national number to report, with where it came from.

    Preference: a valid number from the glyph classifier, then a valid one read from the VL text, then
    whichever attempt has 14 digits (flagged invalid), then the glyph attempt. Nothing is invented."""
    candidates = [(glyph, "glyph classifier"), (from_text, "PaddleOCR-VL text")]
    for nid, source in candidates:
        if nid.get("valid"):
            chosen = dict(nid, source=source)
            other = from_text if nid is glyph else glyph
            if other.get("valid") and other.get("value") != nid.get("value"):
                chosen["conflict"] = other.get("value")          # two valid but different numbers: review it
            return chosen
    for nid, source in candidates:
        if _has_digits(nid):
            return dict(nid, source=source)
    return dict(glyph if glyph.get("value") else from_text, source="not read")


def build_id_fields(vl_text: str, glyph_number: Dict[str, Any]) -> Dict[str, Any]:
    """Fields dict shaped like the ID template output (``id_summary`` accepts it)."""
    parsed = id_fields_from_text(vl_text)
    nid = pick_number(glyph_number, parsed["national_id"])
    reasons = []
    if not nid.get("valid"):
        reasons.append("national number not valid")
    if nid.get("conflict"):
        reasons.append("the two readers disagree on the number")
    if not parsed["full_name"]:
        reasons.append("name not found")
    if not parsed["address_lines"]:
        reasons.append("address not found")
    return {
        "document_type": "egypt_national_id",
        "full_name": parsed["full_name"],
        "birth_date": nid.get("birth_date", ""),
        "address_lines": parsed["address_lines"],
        "national_id": nid,
        "gender": nid.get("gender", ""),
        "serial": parsed.get("serial", ""),
        "needs_review": bool(reasons),
        "review_reasons": reasons,
    }

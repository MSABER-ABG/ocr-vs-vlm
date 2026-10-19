"""Key fields from PLAIN TEXT, so the Document-type tables work with every OCR engine.

ABG OCR reads fields from word boxes (more accurate). Tesseract / Easy OCR / Paddle OCR /
PaddleOCR-VL only give text, so this module recovers the same fields from the text with simple
rules and returns dicts shaped like the ID / cheque template output (``id_summary`` and
``cheque_summary`` accept them). Nothing is invented: a field that is not found stays empty.
"""
import difflib
import re
from typing import Dict, List

from .cheque import BANK_NAMES_AR, norm_ar, parse_amount, parse_date
from .egypt_id import validate_national_id

_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
PAGE_BREAK = "--- Page Break ---"


def to_ascii_digits(text: str) -> str:
    return text.translate(_MAP)


def split_pages(text: str) -> List[str]:
    return [p.strip() for p in text.split(PAGE_BREAK) if p.strip()] or [""]


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


_SERIAL_RE = re.compile(r"[A-Za-z]{1,3}\s?\d{6,9}")
_HEADER_WORDS = ("جمهوري", "مصر", "العربي", "بطاق", "تحقيق", "الشخصي", "وزار", "الداخلي")
_HEADER_FULL = ("جمهورية مصر العربية", "بطاقة تحقيق الشخصية")
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LETTER_RE = re.compile("[ء-يٱ-ۓ]")                          # Arabic LETTERS (not digits)
_SUBTITLE_RE = re.compile(r"بطاق\S*\s+\S*يق\s+ال\S*")                            # "بطاقة تحقيق الشخصية", tolerant
_TITLE_RE = re.compile(r"جمهور\S*\s+مصر\s+العربي\S*")                            # "جمهورية مصر العربية", tolerant


def _fix_digit_dots(text: str) -> str:
    """The card's printed zero is a dot: a dot touching a digit is a zero."""
    return re.sub(r"(?<=[٠-٩0-9])\.|\.(?=[٠-٩0-9])", "٠", text)


def _norm_letters(t: str) -> str:
    return t.replace("ی", "ي").replace("ى", "ي").replace("ک", "ك")


def _is_header(line: str) -> bool:
    n = _norm_letters(line)
    if any(w in n for w in _HEADER_WORDS):
        return True
    return any(difflib.SequenceMatcher(None, n, h).ratio() >= 0.6 for h in _HEADER_FULL)


def id_fields_from_text(text: str) -> dict:
    lines = _lines(text)
    ascii_lines = [to_ascii_digits(ln) for ln in lines]

    number = ""
    for ln in ascii_lines:                           # a 14-digit run, possibly split by spaces
        m = re.search(r"(?<!\d)(\d[\d ]{12,20}\d)(?!\d)", ln)
        if m and len(m.group(1).replace(" ", "")) == 14:
            number = m.group(1).replace(" ", "")
            break
    if not number:                                   # digits split by dots/spaces/letters inside one line
        best = ""
        for ln in ascii_lines:
            if _SERIAL_RE.search(ln.replace(" ", "")):
                continue                              # the card serial (e.g. GU8641199) is not the number
            d = re.sub(r"\D", "", ln)
            if len(d) > len(best):
                best = d
        number = best if len(best) >= 10 else ""      # too short = not the number; never show the serial
    nid = validate_national_id(number) if number else {"value": "", "valid": False, "errors": ["not found"]}

    # Everything up to the printed subtitle is header (the decorative title is often garbled); a line is kept
    # only if it has Arabic LETTERS, so the printed date / number read as digits or look-alikes are dropped.
    start = next((i for i, l in enumerate(lines) if _SUBTITLE_RE.search(_norm_letters(l))), -1) + 1
    text_lines = []
    for ln in lines[start:]:
        ln = _SERIAL_RE.sub("", _TITLE_RE.sub("", _SUBTITLE_RE.sub("", _norm_letters(ln)))).strip()
        if _LETTER_RE.search(ln) and not _is_header(ln):
            text_lines.append(_fix_digit_dots(ln))
    # card layout: first name, rest of the name, then the address lines
    name = " ".join(text_lines[:2])
    address = text_lines[2:4]
    serial = next((m.group(0).replace(" ", "") for m in (_SERIAL_RE.search(a.replace(" ", "")) for a in ascii_lines) if m), "")
    return {"full_name": name, "birth_date": nid.get("birth_date", ""),
            "address_lines": address, "national_id": nid, "serial": serial}


def cheque_fields_from_text(text: str) -> dict:
    lines = _lines(text)
    ascii_text = to_ascii_digits(text)

    date_iso = ""
    for ln in ascii_text.splitlines():
        date_iso = parse_date(ln) or ""
        if date_iso:
            break

    amount, amount_digits = None, ""
    for ln in ascii_text.splitlines():
        if re.search(r"\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{4}", ln):
            continue                                  # skip the date line
        a = parse_amount(ln)
        if a is not None and (amount is None or a > amount):
            amount, amount_digits = a, ln.strip()

    bank_ar, bank_en = "", ""
    for ln in lines:
        n = norm_ar(ln)
        hit = [v for v in BANK_NAMES_AR if len(norm_ar(v)) >= 6 and norm_ar(v) in n]
        if hit:
            bank_ar = max(hit, key=len)
            break
    if not bank_ar:
        bank_en = next((ln for ln in lines if "bank" in ln.lower()), "")

    return {"payee": "", "amount_value": amount, "currency": "", "amount_digits": amount_digits,
            "date": date_iso, "date_as_printed": "", "bank_name_ar": bank_ar, "bank_name_en": bank_en,
            "issues": ["Fields were extracted from plain text only - the customer name needs ABG OCR or a VLM"]}


def fields_from_text(doc_type: str, text: str) -> Dict[str, dict]:
    """One fields dict per page (pages split on the page-break marker); {} for 'General'."""
    fn = {"Egyptian ID card": id_fields_from_text, "Cheque": cheque_fields_from_text}.get(doc_type)
    return [fn(p) for p in split_pages(text)] if fn else []

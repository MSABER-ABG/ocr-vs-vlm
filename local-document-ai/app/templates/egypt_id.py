"""Egyptian national ID card (بطاقة تحقيق الشخصية) reader.

card detect + warp  ->  line OCR for names/address/serial
                    ->  glyph-by-glyph read of the 14-digit number, then VALIDATE it
                        (length, century, real birth date, governorate code).
Birth date and gender are derived from the number itself, which is authoritative.
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from PIL import Image

from app.layout import Line
from app.pipeline import DocumentAI
from app.templates.card import CARD_H, CARD_W, extract_card
from app.templates.digits import read_digit_line, to_western_digits

GOVERNORATES: Dict[str, str] = {
    "01": "Cairo", "02": "Alexandria", "03": "Port Said", "04": "Suez",
    "11": "Damietta", "12": "Dakahlia", "13": "Sharqia", "14": "Qalyubia",
    "15": "Kafr El Sheikh", "16": "Gharbia", "17": "Monufia", "18": "Beheira",
    "19": "Ismailia", "21": "Giza", "22": "Beni Suef", "23": "Fayoum",
    "24": "Minya", "25": "Assiut", "26": "Sohag", "27": "Qena", "28": "Aswan",
    "29": "Luxor", "31": "Red Sea", "32": "New Valley", "33": "Matrouh",
    "34": "North Sinai", "35": "South Sinai", "88": "Born outside Egypt",
}

_ARABIC = re.compile(r"[؀-ۿ]")
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")

# Printed identically on every card. Decorative calligraphy defeats OCR, so the
# template supplies the text instead of guessing it.
CARD_TITLE = "جمهورية مصر العربية"
CARD_SUBTITLE = "بطاقة تحقيق الشخصية"
_SERIAL = re.compile(r"[A-Z]{1,3}\s?\d{6,9}")

# Candidate bands (x1, y1, x2, y2 as card fractions) for the digit line. The first
# is the standard position; the rest cover cards that are shifted/cropped.
_NUMBER_ZONES = [
    (0.38, 0.70, 1.00, 0.90),
    (0.38, 0.66, 1.00, 0.86),
    (0.38, 0.74, 1.00, 0.94),
    (0.30, 0.70, 1.00, 0.90),
    (0.38, 0.62, 1.00, 0.82),
]


def validate_national_id(digits: str) -> dict:
    """Structural validation of the 14-digit Egyptian national number.

    Layout: C YYMMDD GG SSSS K  (century, birth date, governorate, sequence, check).
    The check digit's algorithm is not publicly specified, so it is not verified.
    """
    out = {"value": digits, "valid": False, "errors": []}
    if len(digits) != 14 or not digits.isdigit():
        out["errors"].append(f"expected 14 digits, got {len(digits)}")
        return out

    century = {"2": 1900, "3": 2000}.get(digits[0])
    if century is None:
        out["errors"].append(f"unknown century digit {digits[0]!r} (expected 2 or 3)")
    else:
        try:
            born = date(century + int(digits[1:3]), int(digits[3:5]), int(digits[5:7]))
            if born > date.today():
                out["errors"].append("birth date is in the future")
            else:
                out["birth_date"] = born.isoformat()
        except ValueError:
            out["errors"].append("birth date is not a real calendar date")

    gov = digits[7:9]
    if gov in GOVERNORATES:
        out["governorate_code"] = gov
        out["birth_governorate"] = GOVERNORATES[gov]
    else:
        out["errors"].append(f"unknown governorate code {gov}")

    out["gender"] = "male" if int(digits[12]) % 2 == 1 else "female"
    out["valid"] = not out["errors"]
    return out


@dataclass
class IDCardResult:
    fields: dict = field(default_factory=dict)
    lines: List[Line] = field(default_factory=list)      # raw OCR lines
    text_lines: List[str] = field(default_factory=list)  # cleaned, top-to-bottom plain text


class EgyptIDReader:
    def __init__(self, docai: DocumentAI):
        self.docai = docai
        # Digit-by-digit reading uses the PP-OCRv4 recognizer, which is the one
        # validated on the card's digit font, even when the text engine is v5.
        if hasattr(docai.ocr, "_readers"):
            self._ar = docai.ocr.get_reader("ar")
        else:
            from app.ocr.paddle_v4 import make_reader
            self._ar = make_reader("ar", docai.config)

    # ---------------------------------------------------------------- number
    def read_number(self, card: Image.Image) -> dict:
        """Try candidate bands until one yields a structurally valid national ID."""
        best: Optional[dict] = None
        for zone in _NUMBER_ZONES:
            x1, y1, x2, y2 = zone
            band = card.crop((int(x1 * CARD_W), int(y1 * CARD_H), int(x2 * CARD_W), int(y2 * CARD_H)))
            digits, glyphs = read_digit_line(band, self._ar)
            result = validate_national_id(digits)
            result["glyph_confidence"] = round(
                sum(g.confidence for g in glyphs) / max(len(glyphs), 1), 3
            )
            if result["valid"]:
                return result
            # keep the closest-to-14-digits attempt for reporting
            if best is None or abs(len(digits) - 14) < abs(len(best["value"]) - 14):
                best = result
        return best or validate_national_id("")

    @staticmethod
    def number_from_lines(lines: List[Line]) -> Optional[dict]:
        """A validated 14-digit number that the text engine read as a whole line."""
        for ln in lines:
            digits = to_western_digits(ln.text)
            if len(digits) == 14:
                result = validate_national_id(digits)
                if result["valid"]:
                    result["source"] = "line"
                    return result
        return None

    # ----------------------------------------------------------------- text
    @staticmethod
    def _text_block(lines: List[Line]) -> List[Line]:
        """Arabic lines in the right-hand text block, between the title and the number."""
        block = []
        for ln in lines:
            cx = (ln.bbox[0] + ln.bbox[2]) / 2
            cy = (ln.bbox[1] + ln.bbox[3]) / 2
            if cx > 0.45 * CARD_W and 0.31 * CARD_H < cy < 0.72 * CARD_H and _ARABIC.search(ln.text):
                block.append(ln)
        return sorted(block, key=lambda l: l.bbox[1])

    @staticmethod
    def _serial(lines: List[Line]) -> str:
        for ln in lines:
            m = _SERIAL.search(ln.text.replace(" ", ""))
            if m and (ln.bbox[0] + ln.bbox[2]) / 2 < 0.6 * CARD_W:
                return m.group(0).replace(" ", "")
        return ""

    @staticmethod
    def _clean_text_lines(lines: List[Line], number: dict) -> List[str]:
        """Plain text of the card, top to bottom.

        - the decorative header/subtitle come from the template (fixed printed text);
        - once the national number is validated, the printed birth date (hidden under
          a hologram, so OCR garbles it) is rebuilt from the number, and the number
          itself is added. Nothing is invented when validation fails.
        """
        items = []
        valid = number["valid"]
        for ln in lines:
            x1, _, _, _ = ln.bbox
            cy = (ln.bbox[1] + ln.bbox[3]) / 2
            if cy < 0.30 * CARD_H:
                continue                                    # header zone -> constants below
            no_letters = not any(c.isalpha() for c in ln.text)
            if valid and 0.60 * CARD_H < cy < 0.90 * CARD_H and _DIGIT.search(ln.text)                     and not _SERIAL.search(ln.text.replace(" ", ""))                     and (x1 < 0.30 * CARD_W or no_letters):
                continue          # garbled printed date / number fragments -> rebuilt below
            items.append((cy, ln.text))

        items.append((0.00 * CARD_H, CARD_TITLE))
        items.append((0.02 * CARD_H, CARD_SUBTITLE))
        if valid:
            y, m, d = number["birth_date"].split("-")
            items.append((0.78 * CARD_H, f"{int(y)}/{int(m)}/{int(d)}"))
            items.append((0.85 * CARD_H, number["value"]))
        return [t for _, t in sorted(items, key=lambda it: it[0])]

    # ------------------------------------------------------------------ main
    def read(self, image: Image.Image) -> IDCardResult:
        card = extract_card(image)
        cfg = self.docai.config
        saved = cfg.max_upscale
        cfg.max_upscale = 1.0                      # the warped card is already large
        try:
            lines = self.docai.process_pages([card]).pages[0].lines
        finally:
            cfg.max_upscale = saved

        block = self._text_block(lines)
        texts = [ln.text for ln in block]
        number = self.read_number(card)
        if not number["valid"]:
            number = self.number_from_lines(lines) or number

        fields = {
            "document_type": "egypt_national_id",
            "first_name": texts[0] if len(texts) > 0 else "",
            "rest_of_name": texts[1] if len(texts) > 1 else "",
            "address_lines": texts[2:],
            "national_id": number,
            "birth_date": number.get("birth_date", ""),
            "gender": number.get("gender", ""),
            "serial": self._serial(lines),
        }
        fields["full_name"] = " ".join(t for t in (fields["first_name"], fields["rest_of_name"]) if t)
        fields["needs_review"] = (not number["valid"]) or len(texts) < 4 or not fields["serial"]
        return IDCardResult(fields=fields, lines=lines,
                            text_lines=self._clean_text_lines(lines, number))

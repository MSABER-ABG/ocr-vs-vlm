"""Bank cheque reader (bilingual Arabic/English cheques).

A cheque is a pre-printed FORM: fixed labels + a few variable values. So instead of
trusting a free-text read of the whole page we find each field by its CONTENT and
its POSITION relative to the printed labels:

  * date            - a real calendar date next to the "Date" label
  * amount (digits) - the number next to the currency (EGP ...)
  * amount (words)  - the text made of number words, English OR Arabic, possibly on
                      several lines
  * payee           - the free text on the "Pay to" row (Arabic or Latin); if the
                      labels are not usable, the first free text under the date row
  * signatory       - the text next to "Signatory Name"

Printed Arabic labels are snapped to a closed vocabulary (fixed text). The amount in
words is cross-checked against the digits. Anything that fails a check is reported in
``issues`` and sets ``needs_review``; nothing is invented.
"""
import difflib
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

from app.layout import Line, sort_reading_order
from app.layout.reading_order import _kind
from app.ocr.base import OCRResult
from app.pipeline import DocumentAI

# ---------------------------------------------------------------- vocabularies
# Fixed printed Arabic labels / stamps found on Egyptian cheques (extend per bank).
ARABIC_LABELS = [
    "التاريخ", "إدفعوا بموجب هذا الشيك", "إدفعوا بموجب هذا الشيك لأمر", "إدفعوا بموجب", "هذا الشيك",
    "لأمر", "مبلغاً وقدره", "المبلغ", "اسم الساحب", "اسم الموقع", "التوقيع", "شيك رقم", "شيك",
    "صرف للمستفيد الأول", "اقفل بشرط عدم الصرف نقداً",
]
# Seed list; extend it per customer. Only used to snap near-exact matches (>= 0.8).
BANK_NAMES_AR = [
    "البنك العربي الأفريقي الدولي", "البنك الأهلي المصري", "بنك مصر", "البنك التجاري الدولي",
    "بنك القاهرة", "بنك الإسكندرية", "بنك قطر الوطني الأهلي", "البنك الأهلي الكويتي",
    "بنك الإمارات دبي الوطني", "بنك فيصل الإسلامي المصري", "المصرف المتحد", "بنك التعمير والإسكان",
]
# English printed labels (normalised to lowercase letters only).
EN_LABELS: Dict[str, List[str]] = {
    "date": ["date"],
    "payee": ["toorder", "payto", "paytotheorderof", "paybythischeck", "paybythischeque",
              "againstthischeque", "againstthischeck", "againstthischoque"],
    "sum_of": ["thesumof"],
    "name": ["name", "signatoryname", "signaturename"],
    "signature": ["signature"],
    "sum": ["sum"],
}
CURRENCIES = {"egp", "le", "usd", "eur", "sar", "aed", "kwd", "gbp"}
# Printed stamps / security text that must never be mistaken for a field value.
_NOISE_EN = ("acpayee", "payeeonly", "accountpayee", "securityfeatures", "detailsontheback", "included")
_NOISE_AR = ("للمستفيد", "لايصرف", "نقدا")

# ----------------------------------------------------------- number words (EN)
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("zero ten twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_IGNORABLE = {"and", "only", "egyptian", "pound", "pounds", "egp", "le", "piastres", "piasters", "no", "dollars"}
_NUMBER_VOCAB = list(_UNITS) + list(_TENS) + list(_SCALES) + ["hundred"]

# ----------------------------------------------------------- number words (AR)
# Normalised spelling (see norm_ar): ا for أإآ, ي for ى, ه for ة.
_AR_UNITS = {
    "واحد": 1, "احد": 1, "اثنان": 2, "اثنين": 2, "اثنا": 2, "اثني": 2, "ثلاثه": 3, "ثلاث": 3,
    "اربعه": 4, "اربع": 4, "خمسه": 5, "خمس": 5, "سته": 6, "ست": 6, "سبعه": 7, "سبع": 7,
    "ثمانيه": 8, "ثماني": 8, "ثمان": 8, "تسعه": 9, "تسع": 9, "عشره": 10, "عشر": 10,
}
_AR_TENS = {
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "اربعون": 40, "اربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
_AR_HUNDREDS = {
    "مائه": 100, "مئه": 100, "ميه": 100, "مايه": 100,
    "مائتان": 200, "مئتان": 200, "مائتين": 200, "مئتين": 200, "ميتين": 200,
    "ثلاثمائه": 300, "ثلاثمئه": 300, "اربعمائه": 400, "اربعمئه": 400, "خمسمائه": 500, "خمسمئه": 500,
    "ستمائه": 600, "ستمئه": 600, "سبعمائه": 700, "سبعمئه": 700,
    "ثمانمائه": 800, "ثمانيمائه": 800, "ثمانمئه": 800, "تسعمائه": 900, "تسعمئه": 900,
}
_AR_SCALES = {"الف": 1_000, "الاف": 1_000, "مليون": 1_000_000, "ملايين": 1_000_000,
              "مليار": 1_000_000_000, "مليارات": 1_000_000_000}
_AR_DUAL = {"الفان": 2_000, "الفين": 2_000, "مليونان": 2_000_000, "مليونين": 2_000_000,
            "ملياران": 2_000_000_000, "مليارين": 2_000_000_000}
_AR_IGNORABLE = {"فقط", "لا", "غير", "جنيه", "جنيها", "جنيهات", "مصري", "مصريا", "مصريه", "قرش", "قروش",
                 "ريال", "دولار", "درهم", "دينار", "ومائه"}
# Very common names that are close in spelling to number words ("احمد" ~ "احد"): exact match only.
_AR_NAME_GUARD = {"احمد", "محمد", "محمود", "حمد", "عمر", "عمرو", "سعد", "ماجد", "اشرف", "ايمن", "امير"}
_AR_ALL = {**_AR_UNITS, **_AR_TENS, **_AR_HUNDREDS, **_AR_SCALES, **_AR_DUAL}


# ------------------------------------------------------------------ utilities
def norm_ar(s: str) -> str:
    s = re.sub("[ً-ٟـ]", "", s)                       # diacritics, tatweel
    s = re.sub("[أإآ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w ]", "", s).strip()


def snap(text: str, vocab: List[str], cutoff: float) -> Tuple[str, bool]:
    """Replace ``text`` by the closest vocabulary entry when similarity >= cutoff."""
    n = norm_ar(text)
    if len(n) < 3:
        return text, False
    best, score = text, 0.0
    for v in vocab:
        r = difflib.SequenceMatcher(None, n, norm_ar(v)).ratio()
        if r > score:
            best, score = v, r
    return (best, True) if score >= cutoff else (text, False)


def words_to_number(text: str) -> Tuple[Optional[int], int]:
    """English amount in words -> integer. Returns ``(value, unrecognised_token_count)``."""
    total = current = unknown = 0
    seen = False
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok not in _NUMBER_VOCAB and tok not in _IGNORABLE:
            near = difflib.get_close_matches(tok, _NUMBER_VOCAB, n=1, cutoff=0.8)   # OCR typos
            if not near:
                unknown += 1
                continue
            tok = near[0]
        if tok in _UNITS:
            current += _UNITS[tok]; seen = True
        elif tok in _TENS:
            current += _TENS[tok]; seen = True
        elif tok == "hundred":
            current = (current or 1) * 100; seen = True
        elif tok in _SCALES:
            total += (current or 1) * _SCALES[tok]; current = 0; seen = True
    return (total + current if seen else None), unknown


def _ar_token(tok: str) -> Optional[str]:
    """Map one (normalised) Arabic token to a number-word key, tolerating OCR errors."""
    if tok in _AR_NAME_GUARD:
        return None
    forms = [tok]
    if tok.startswith("و") and len(tok) > 2 and tok != "واحد":          # conjunction "and"
        forms.append(tok[1:])
    for f in forms:
        if f in _AR_ALL:
            return f
    best, score = None, 0.0
    for f in forms:
        for key in _AR_ALL:
            if abs(len(key) - len(f)) > 2:
                continue
            r = difflib.SequenceMatcher(None, f, key).ratio()
            if r > score:
                best, score = key, r
    return best if score >= 0.8 else None


def arabic_words_to_number(text: str) -> Tuple[Optional[int], int]:
    """Arabic amount in words -> integer. Returns ``(value, unrecognised_token_count)``."""
    total = current = unknown = 0
    seen = False
    for raw in norm_ar(text).split():
        key = _ar_token(raw)
        if key is None:
            bare = raw[1:] if raw.startswith("و") and len(raw) > 2 else raw
            if bare in _AR_IGNORABLE or bare.startswith(("مصري", "جنيه", "جنيها", "مصريا")):
                continue
            unknown += 1
            continue
        if key in _AR_UNITS:
            current += _AR_UNITS[key]; seen = True
        elif key in _AR_TENS:
            current += _AR_TENS[key]; seen = True
        elif key in _AR_HUNDREDS:
            current += _AR_HUNDREDS[key]; seen = True
        elif key in _AR_DUAL:
            total += _AR_DUAL[key]; seen = True
        elif key in _AR_SCALES:
            total += (current or 1) * _AR_SCALES[key]; current = 0; seen = True
    return (total + current if seen else None), unknown


def number_word_stats(text: str) -> Tuple[int, int, int]:
    """``(number_words, filler_words, total_tokens)`` for English + Arabic text."""
    n_num = n_fill = 0
    toks = re.findall(r"[A-Za-z]+|[؀-ۿ]+", text)
    for tok in toks:
        if re.match(r"[A-Za-z]", tok):
            t = tok.lower()
            if t in _NUMBER_VOCAB or difflib.get_close_matches(t, _NUMBER_VOCAB, n=1, cutoff=0.85):
                n_num += 1
            elif t in _IGNORABLE:
                n_fill += 1
        else:
            nt = norm_ar(tok)
            if _ar_token(nt) is not None:
                n_num += 1
            else:
                bare = nt[1:] if nt.startswith("و") and len(nt) > 2 else nt
                if bare in _AR_IGNORABLE or bare.startswith(("مصري", "جنيه", "جنيها")):
                    n_fill += 1
    return n_num, n_fill, len(toks)


def count_number_words(text: str) -> int:
    return number_word_stats(text)[0]


def looks_like_amount_words(text: str) -> bool:
    n_num, n_fill, total = number_word_stats(text)
    return n_num >= 2 and total > 0 and (n_num + n_fill) / total >= 0.7


_DATE = re.compile(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})")


def parse_date(text: str) -> Optional[str]:
    """Day-first date (Egyptian convention) -> ISO string, or None if not a real date."""
    m = _DATE.search(text)
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_amount(text: str) -> Optional[float]:
    m = _AMOUNT.search(text.replace(" ", ""))
    if not m or sum(c.isdigit() for c in m.group(0)) < 3:
        return None
    return float(m.group(0).replace(",", ""))


_PRINTED_EN = re.compile(
    r"(?i)(against\s+th[il1]s\s+ch[eo][qc]?ue|against\s+th[il1]s\s+che[qc]k|pay\s+to(\s+the\s+order\s+of)?"
    r"|to\s+order|pay\s+by\s+th[il1]s\s+che[qc]k)")


def strip_printed_phrases(text: str) -> str:
    """Remove fixed printed labels that OCR merged into a value box."""
    return _PRINTED_EN.sub(" ", text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_#]", " ", text)).strip(" '\"`")


def _enorm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _has_arabic(w: OCRResult) -> bool:
    return _kind(w.text) == "ar"


def _cy(w: OCRResult) -> float:
    return (w.bbox[1] + w.bbox[3]) / 2


def _h(w: OCRResult) -> float:
    return w.bbox[3] - w.bbox[1]


def _same_row(a: OCRResult, b: OCRResult) -> bool:
    """True when two boxes share a text row: vertical overlap >= half the smaller height."""
    overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    return overlap >= 0.5 * min(_h(a), _h(b))


def _label_match(box: OCRResult, variants: List[str]) -> float:
    if _has_arabic(box):
        return 0.0
    n = _enorm(box.text)
    best = 0.0
    for v in variants:
        if abs(len(n) - len(v)) <= 3 and len(n) >= 3:
            best = max(best, difflib.SequenceMatcher(None, n, v).ratio())
    return best if best >= 0.85 else 0.0


def find_labels(boxes: List[OCRResult], key: str) -> List[OCRResult]:
    return [b for b in boxes if _label_match(b, EN_LABELS[key])]


def find_label(boxes: List[OCRResult], key: str) -> Optional[OCRResult]:
    found = find_labels(boxes, key)
    return max(found, key=lambda b: _label_match(b, EN_LABELS[key]) * b.confidence) if found else None


def _is_noise(text: str) -> bool:
    e = _enorm(text)
    if e in {"cheque", "check", "chequeat", "chequeatfreecheques"} or e.startswith("chequeat"):
        return True
    if any(p in e for p in _NOISE_EN):
        return True
    n = norm_ar(text)
    return any(p in n for p in _NOISE_AR)


# --------------------------------------------------------------------- reader
@dataclass
class ChequeResult:
    fields: dict = field(default_factory=dict)
    text_lines: List[str] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)


def _split_runs(line: Line) -> List[str]:
    """One output row per script run, so English and Arabic of the same row are separate lines."""
    groups, cur, cur_kind = [], [], None
    for w in line.words:
        neutral = not any(c.isalpha() for c in w.text)
        k = _kind(w.text)
        if cur and not neutral and k != cur_kind:
            groups.append(" ".join(cur)); cur = []
        cur.append(w.text)
        if not neutral:
            cur_kind = k
    if cur:
        groups.append(" ".join(cur))
    return groups


def _reading_text(boxes: List[OCRResult]) -> str:
    """Join boxes of one value: rows top-to-bottom, Arabic right-to-left, Latin left-to-right."""
    rows: List[List[OCRResult]] = []
    for b in sorted(boxes, key=_cy):
        if rows and _same_row(rows[-1][0], b):
            rows[-1].append(b)
        else:
            rows.append([b])
    out = []
    for r in rows:
        r.sort(key=lambda b: -b.bbox[2] if _has_arabic(b) else b.bbox[0])
        out.append(" ".join(b.text for b in r))
    return _clean(" ".join(out))


def extract_fields(boxes: List[OCRResult], image_height: float,
                   label_ids: Optional[Set[int]] = None) -> Tuple[dict, List[str]]:
    """Pure function on OCR boxes -> (fields, issues). No models involved.

    ``label_ids`` = ids of boxes already recognised as printed Arabic labels."""
    issues: List[str] = []
    label_ids = set(label_ids or ())
    for b in boxes:                                    # printed Arabic labels (fixed text)
        if _has_arabic(b) and snap(b.text, ARABIC_LABELS, 0.7)[1]:
            label_ids.add(id(b))
    en_labels = {k: find_labels(boxes, k) for k in EN_LABELS}
    for lst in en_labels.values():
        label_ids |= {id(b) for b in lst}
    noise_ids = {id(b) for b in boxes if _is_noise(b.text)}
    printed = label_ids | noise_ids

    # ---- date -------------------------------------------------------------------------
    date_label = max(en_labels["date"], key=lambda b: b.confidence, default=None)
    date_boxes = [b for b in boxes if _DATE.search(b.text)]
    if date_label and date_boxes:
        date_boxes.sort(key=lambda b: abs(_cy(b) - _cy(date_label)) + abs(b.bbox[0] - date_label.bbox[2]) * 0.2)
    date_box = date_boxes[0] if date_boxes else None
    date_raw = date_box.text.strip() if date_box else ""
    date_iso = parse_date(date_raw) if date_raw else None
    if not date_raw:
        issues.append("date not found")
    elif not date_iso:
        issues.append(f"date is not a valid calendar date: {date_raw!r}")

    # ---- currency + amount in digits ------------------------------------------------------
    currency_box = next((b for b in boxes if _enorm(b.text) in CURRENCIES), None)
    amount_boxes = [b for b in boxes if parse_amount(b.text) is not None and not _DATE.search(b.text)
                    and not _has_arabic(b) and id(b) not in printed]
    if currency_box and amount_boxes:
        amount_boxes.sort(key=lambda b: abs(_cy(b) - _cy(currency_box)) + abs(b.bbox[0] - currency_box.bbox[2]) * 0.2)
    amount_box = amount_boxes[0] if amount_boxes else None
    amount_value = parse_amount(amount_box.text) if amount_box else None
    if amount_value is None:
        issues.append("amount in digits not found")

    # ---- amount in words: boxes made of number words (English or Arabic) --------------------
    sum_label = max(en_labels["sum_of"], key=lambda b: b.confidence, default=None)
    word_boxes = [b for b in boxes if id(b) not in printed and b is not amount_box
                  and not _DATE.search(b.text) and looks_like_amount_words(b.text)]
    words_text, words_value, unknown = "", None, 0
    if word_boxes:
        # keep boxes that sit in one compact block (same side of the page, near each other)
        anchor = max(word_boxes, key=lambda b: count_number_words(b.text))
        block = [b for b in word_boxes
                 if abs(_cy(b) - _cy(anchor)) <= 3.5 * max(_h(anchor), 1)
                 and abs(((b.bbox[0] + b.bbox[2]) / 2) - ((anchor.bbox[0] + anchor.bbox[2]) / 2)) <= 0.7 *
                 max(anchor.bbox[2] - anchor.bbox[0], 120)]
        words_text = _reading_text(block)
        if re.search(r"[A-Za-z]{3}", words_text) and not re.search(r"[؀-ۿ]{3}", words_text):
            words_value, unknown = words_to_number(words_text)
        else:
            words_value, unknown = arabic_words_to_number(words_text)
    if words_value is None:
        issues.append("amount in words not found")

    amount_match = None
    if words_value is not None and amount_value is not None:
        amount_match = int(amount_value) == words_value
        if not amount_match:
            extra = f" ({unknown} unrecognised word(s))" if unknown else ""
            issues.append(f"amount mismatch: words = {words_value:,}, digits = {int(amount_value):,}{extra}")

    # ---- payee ------------------------------------------------------------------------------
    used = {id(b) for b in word_boxes} | {id(date_box)} if date_box else {id(b) for b in word_boxes}
    if amount_box:
        used.add(id(amount_box))
    if currency_box:
        used.add(id(currency_box))
    top_limit = sum_label.bbox[1] if sum_label else (min((b.bbox[1] for b in word_boxes), default=image_height))

    def free_text(b: OCRResult) -> bool:
        return (id(b) not in printed and id(b) not in used and b.confidence >= 0.5
                and any(c.isalpha() for c in b.text) and "bank" not in b.text.lower())

    payee_boxes: List[OCRResult] = []
    placeholder = next((_clean(b.text) for b in boxes if re.fullmatch(r"[xX]{4,}", _clean(b.text))), "")
    pay_labels = en_labels["payee"]
    if pay_labels:
        left = min(b.bbox[2] for b in pay_labels)
        top = min(b.bbox[1] for b in pay_labels) - 1.0 * max(_h(b) for b in pay_labels)
        bottom = max(b.bbox[3] for b in pay_labels) + 0.3 * max(_h(b) for b in pay_labels)
        payee_boxes = [b for b in boxes if free_text(b) and b.bbox[0] >= left - 8
                       and any(_same_row(b, lab) for lab in pay_labels) and _cy(b) < top_limit + 2]
    if not payee_boxes:                                   # fallback: first free text under the date row
        y_from = date_box.bbox[3] if date_box else 0
        cand = sorted((b for b in boxes if free_text(b) and b.bbox[1] >= y_from - 2 and _cy(b) < top_limit),
                      key=_cy)
        if cand:
            first = cand[0]
            payee_boxes = [b for b in cand if _same_row(first, b)]
    cleaned = []
    for b in payee_boxes:
        t = _clean(strip_printed_phrases(b.text))
        if t and not re.fullmatch(r"[xX]{4,}", t):
            cleaned.append(OCRResult(page=b.page, text=t, bbox=b.bbox, confidence=b.confidence))
    payee = _reading_text(cleaned)
    if not payee:
        issues.append("payee not found")

    # ---- signatory / drawer name ----------------------------------------------------------------------
    drawer = ""
    name_lab = max(en_labels["name"], key=lambda b: b.confidence, default=None)
    if name_lab:
        x_limit = currency_box.bbox[0] if currency_box and _same_row(currency_box, name_lab) else float("inf")
        cand = [b for b in boxes if free_text(b) and _same_row(b, name_lab) and b is not name_lab
                and (_has_arabic(b) or b.bbox[0] >= name_lab.bbox[2]) and b.bbox[0] < x_limit]
        drawer = _reading_text(cand)

    # ---- bank ------------------------------------------------------------------------------------------------------
    bank_en = next((_clean(b.text) for b in sorted(boxes, key=lambda b: -b.confidence)
                    if "bank" in b.text.lower() and not _has_arabic(b)), "")
    bank_ar = ""
    for b in boxes:
        if _has_arabic(b) and _cy(b) < 0.45 * image_height:
            nb = norm_ar(b.text)
            contained = [v for v in BANK_NAMES_AR if len(norm_ar(v)) >= 6 and norm_ar(v) in nb]
            if contained:
                bank_ar = max(contained, key=len)
                break
            cand_name, ok = snap(b.text, BANK_NAMES_AR, 0.8)
            if ok:
                bank_ar = cand_name
                break

    fields = {
        "document_type": "cheque",
        "bank_name_en": bank_en,
        "bank_name_ar": bank_ar,
        "date": date_iso or "",
        "date_as_printed": date_raw,
        "payee": payee,
        "payee_blank_line": placeholder,
        "currency": _enorm(currency_box.text).upper() if currency_box else "",
        "amount_digits": amount_box.text.replace("*", "").replace("#", "").strip() if amount_box else "",
        "amount_value": amount_value,
        "amount_words": words_text,
        "amount_words_value": words_value,
        "amount_words_unrecognised": unknown,
        "amount_words_match_digits": amount_match,
        "signatory_name": drawer,
        "issues": issues,
        "needs_review": bool(issues),
    }
    return fields, issues


def straighten(image: Image.Image, engine) -> Tuple[Image.Image, float]:
    """Rotate a tilted photo so text lines are horizontal (needs an engine with skew estimation)."""
    if not hasattr(engine, "estimate_skew"):
        return image, 0.0
    angle = engine.estimate_skew(image)
    if abs(angle) < 1.5:
        return image, 0.0
    rotated = image.convert("RGB").rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return rotated, angle


class ChequeReader:
    def __init__(self, docai: DocumentAI):
        self.docai = docai

    def _read_once(self, image: Image.Image, angle: float) -> ChequeResult:
        page = self.docai.process_pages([image]).pages[0]
        boxes = [w for ln in page.lines for w in ln.words]

        # Snap printed Arabic labels / bank names to the closed vocabulary.
        snapped, label_ids = 0, set()
        for b in boxes:
            if _has_arabic(b):
                new, ok = snap(b.text, BANK_NAMES_AR, 0.8)
                if not ok:
                    new, ok = snap(b.text, ARABIC_LABELS, 0.7)
                    if ok:
                        label_ids.add(id(b))
                if ok and new != b.text:
                    b.text, snapped = new, snapped + 1

        fields, _ = extract_fields(boxes, page.height, label_ids)
        fields["snapped_printed_labels"] = snapped
        fields["rotation_applied_deg"] = round(angle, 1)

        lines = sort_reading_order(boxes, self.docai.config.line_overlap_ratio)
        text_lines = [g for ln in lines for g in _split_runs(ln)]
        return ChequeResult(fields=fields, text_lines=text_lines, lines=lines)

    def read(self, image: Image.Image) -> ChequeResult:
        """Read as-is; if checks fail and the photo is tilted, also try it straightened and
        keep whichever result passes more checks (rotating blurs small text, so only when needed)."""
        best = self._read_once(image, 0.0)
        if best.fields["issues"]:
            rotated, angle = straighten(image, self.docai.ocr)
            if angle:
                alt = self._read_once(rotated, angle)
                if len(alt.fields["issues"]) < len(best.fields["issues"]):
                    best = alt
        return best

"""Compare models against a REFERENCE model (e.g. o4-mini), field by field, by correctness.

``compare.py`` only says whether a field was found and looks plausible. Here the reference
model's answer is treated as the truth and every other model is scored per field:

* name / text / address / any free text -> share of the reference's WORDS the model also wrote
  (letters unified: hamza, taa marbuta, yaa, Persian forms, diacritics; Arabic digits -> ascii);
* national ID (nid) -> exact 14 digits (the digit-by-digit match is shown next to it);
* amount -> same number;  date -> same calendar date.

Score of a model = the average of its field scores (only fields the reference actually has).
Pure functions: no UI and no model imports.
"""
import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .cheque import parse_date
from .compare import KINDS, MISSING, OK, WARN, columns_for
from .egypt_id import validate_national_id
from .summary import flatten_fields

GENERAL = "General"
_SKIP_KEYS = ("confidence", "language", "notes", "tables", "text", "_page", "parse_error", "raw_response")

_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_AR_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})


def _ascii(text: Any) -> str:
    return str(text if text is not None else "").replace("⚠️", "").translate(_AR_DIGITS)


def _digits(text: Any) -> str:
    return re.sub(r"\D", "", _ascii(text))


def words(text: Any) -> List[str]:
    """Normalised words of a text (Arabic letter variants unified)."""
    v = _ascii(text).lower()
    v = re.sub(r"[ً-ٟـ]", "", v)                           # diacritics / tatweel
    v = re.sub(r"[أإآٱ]", "ا", v)
    v = v.replace("ى", "ي").replace("ی", "ي").replace("ة", "ه").replace("ک", "ك").replace("ؤ", "و").replace("ئ", "ي")
    return [w for w in re.split(r"[\W_]+", v) if w]


def words_score(ref: Any, bag: Counter) -> Tuple[float, str]:
    rc = Counter(words(ref))
    total = sum(rc.values())
    if not total:
        return 100.0, "no words"
    matched = sum(min(c, bag.get(w, 0)) for w, c in rc.items())
    return 100.0 * matched / total, f"{matched}/{total} words"


def _number(text: Any) -> Optional[float]:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", _ascii(text))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _iso_date(text: Any) -> str:
    t = _ascii(text).strip()
    m = re.search(r"(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    try:
        return parse_date(t) or _digits(t)
    except Exception:
        return _digits(t)


def digits_score(r: str, v: str) -> Tuple[float, str]:
    """Digit-by-digit match (like word-by-word for text): share of the reference's digits the model got right.

    Same position counts; if the model dropped/added a digit, the best alignment is used, so one slip
    does not turn every following digit into a mistake."""
    if not r:
        return 0.0, "0/0 digits"
    pos = sum(1 for a, b in zip(r, v) if a == b)
    aligned = sum(blk.size for blk in difflib.SequenceMatcher(None, r, v, autojunk=False).get_matching_blocks())
    hit = min(len(r), max(pos, aligned))
    return 100.0 * hit / len(r), f"{hit}/{len(r)} digits"


def value_score(kind: str, ref: Any, val: Any) -> Tuple[float, str]:
    """(score 0-100, short detail) of one model value against the reference value."""
    if kind == "nid":
        return digits_score(_digits(ref), _digits(val))
    if kind == "date":
        return digits_score(_digits(_iso_date(ref)), _digits(_iso_date(val)))        # YYYYMMDD, digit by digit
    if kind == "amount":
        r, v = _number(ref), _number(val)
        ok = r is not None and v is not None and abs(r - v) < 0.005
        return (100.0 if ok else 0.0), ("same amount" if ok else "amount differs")
    return words_score(ref, Counter(words(val)))


NOISE_ON, NOISE_OFF = chr(0xE000), chr(0xE001)      # html_table turns these into a highlight


def mark_noise(ref: Any, val: Any) -> Tuple[str, int]:
    """The model's value with the words that are NOT in the reference highlighted; and how many there are."""
    left = Counter(words(ref))
    out, extra = [], 0
    for tok in str(val or "").replace("⚠️", "").split():
        ws = words(tok)
        if ws and all(left.get(w, 0) > 0 for w in ws):
            for w in ws:
                left[w] -= 1
            out.append(tok)
        elif ws:
            extra += 1
            out.append(NOISE_ON + tok + NOISE_OFF)
        else:
            out.append(tok)
    return " ".join(out), extra


def _icon(found: bool, pct: float) -> str:
    if not found:
        return MISSING
    return OK if pct >= 90 else WARN if pct >= 50 else MISSING


def reference_fields(doc_type: str, ref_row: Optional[Dict[str, str]], ref_page: Optional[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """``[(field label, kind, reference value)]`` - template columns, or the reference's own keys for General."""
    cols = columns_for(doc_type)
    if cols and ref_row is not None:
        return [(c, k, str(ref_row.get(c, "") or "")) for c, k in zip(cols, KINDS[doc_type])]
    data = dict(ref_page or {})
    src = data.get("fields") if isinstance(data.get("fields"), dict) else {k: v for k, v in data.items() if k not in _SKIP_KEYS}
    out = []
    for name, value in flatten_fields(src):
        if str(name).split(".")[0].lower() in _SKIP_KEYS:
            continue
        out.append((name, "text", value))
    return out


def entry_bag(entry: Dict[str, Any]) -> Counter:
    """All words a model produced (used for General documents)."""
    return Counter(words(entry.get("text", "")))


def _clip(text: Any, n: int = 120) -> str:
    t = str(text or "").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def compare_to_reference(doc_type: str, fields: List[Tuple[str, str, str]], entries: List[Dict[str, Any]],
                         ref_label: str = "Reference"):
    """``entries``: ``[{"label", "row" (template doc types) or "text" (General), "time"}]``.

    Returns ``(headers_ltr, rows_ltr, averages, best_index, notes)``; the layout matches ``html_table``
    (field column last = right-most, model 1 beside the field)."""
    if not fields or not entries:
        return [], [], [], None, []
    notes: List[str] = []
    scores: List[List[float]] = [[] for _ in entries]
    rows = []
    for name, kind, ref in fields:
        if not str(ref).strip():
            rows.append(["—"] * (len(entries) + 1) + [name + " (empty in reference)"])
            continue
        if kind == "nid":
            d = _digits(ref)
            if len(d) != 14 or not validate_national_id(d)["valid"]:
                notes.append("⚠️ The reference's national ID is not a valid 14-digit number - its own reading may be wrong.")
        cells = [f"{OK} Reference · 100%" + chr(10) + _clip(ref)]
        for i, e in enumerate(entries):
            if "row" in e:
                val = e["row"].get(name, "")
                found = bool(str(val or "").replace("⚠️", "").strip())
                pct, detail = value_score(kind, ref, val) if found else (0.0, "")
            else:
                found = True
                pct, detail = words_score(ref, e.get("_bag") or entry_bag(e))
                found = pct > 0
            scores[i].append(pct)
            word = "Not found" if not found else "Correct" if pct >= 90 else "Partly" if pct >= 50 else "Wrong"
            line = f"{_icon(found, pct)} {word} · {pct:.0f}%" + (f" ({detail})" if detail and found else "")  # noqa
            if "row" in e:
                actual = str(e["row"].get(name, "") or "").replace("⚠️", "").strip()
                if found and kind not in ("nid", "amount", "date"):
                    actual, extra = mark_noise(ref, actual)
                    if extra:
                        line += f" + Noise ({extra} extra)"
            else:                                    # General: the reference words this model also wrote
                bag = e.get("_bag") or entry_bag(e)
                actual = " ".join(w for w in dict.fromkeys(words(ref)) if bag.get(w, 0))
            cells.append(line + (chr(10) + _clip(actual) if actual else ""))
        rows.append(cells[::-1] + [name])
    avgs = [round(sum(s) / len(s), 1) if s else 0.0 for s in scores]
    rows.append((["100% average"] + [f"{a:.0f}% average" for a in avgs])[::-1] + ["Average score"])
    headers = ([ref_label] + [e["label"] for e in entries])[::-1] + ["Field"]
    best = max(range(len(entries)), key=lambda i: (avgs[i], -float(entries[i].get("time") or 0)))
    return headers, rows, avgs, best, list(dict.fromkeys(notes))

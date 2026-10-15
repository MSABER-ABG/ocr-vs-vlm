"""Compare several models on the same document by field STATUS (not by value).

For every key field of a document type, each model gets a status:

    ✅ read and plausible   ⚠️ read but looks wrong / needs review   ❌ not found

plus an "agreement" column telling whether the models returned the same thing, so a client
can choose a model without reading the values. Pure functions, no UI / model imports.
"""
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .egypt_id import validate_national_id
from .summary import CHEQUE_COLUMNS, DOC_CHEQUE, DOC_ID, ID_COLUMNS

OK, WARN, MISSING = "✅", "⚠️", "❌"
KINDS = {DOC_ID: ["name", "date", "text", "nid"],
         DOC_CHEQUE: ["name", "amount", "date", "text"]}

_AR_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_AR_DIGITS.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})


def _ascii(text: str) -> str:
    return str(text or "").translate(_AR_DIGITS)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", _ascii(text))


def field_status(kind: str, value: str) -> Tuple[str, str]:
    """(icon, short description) for one extracted value."""
    value = str(value or "").replace("⚠️", "").strip()
    if not value:
        return MISSING, "Not found"
    if kind == "nid":
        d = _digits(value)
        if len(d) != 14:
            return WARN, f"Wrong length ({len(d)} digits)"
        res = validate_national_id(d)
        if res["valid"]:
            return OK, "Valid 14-digit number"
        return WARN, "Fails validation (" + (res["errors"][0] if res["errors"] else "invalid") + ")"
    if kind == "date":
        try:
            date.fromisoformat(value)
            return OK, "Valid date"
        except ValueError:
            return WARN, "Date unreadable"
    if kind == "amount":
        return (OK, "Amount found") if re.search(r"\d", _ascii(value)) else (WARN, "No digits")
    if kind == "name":
        return (WARN, "Contains digits") if re.search(r"\d", _ascii(value)) else (OK, "Found")
    return OK, "Found"


def _norm(kind: str, value: str) -> str:
    value = str(value or "").replace("⚠️", "")
    if kind in ("nid", "amount", "date"):
        return _digits(value)
    v = re.sub(r"[ً-ٟـ]", "", value)                    # diacritics / tatweel
    v = re.sub(r"[أإآ]", "ا", v).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[\W_]+", "", v.lower())


def agreement(kind: str, values: List[str]) -> str:
    """Do the models agree on this field? (without showing the value)"""
    present = [_norm(kind, v) for v in values if str(v or "").strip()]
    present = [p for p in present if p]
    if len(present) < 2:
        return "—"
    best = max(present.count(p) for p in set(present))
    if best == len(present):
        return "🤝 All agree"
    return f"{WARN} {best} of {len(present)} agree"


def columns_for(doc_type: str) -> Optional[List[str]]:
    return {DOC_ID: ID_COLUMNS, DOC_CHEQUE: CHEQUE_COLUMNS}.get(doc_type)


def compare(doc_type: str, entries: List[Dict[str, Any]]):
    """``entries``: ``[{"label": str, "row": {column: value}, "time": float}, ...]``.

    Returns ``(headers_ltr, rows_ltr, scores, best_index)`` in the layout ``html_table`` expects
    (field column last = right-most, one column per model, agreement next to the field)."""
    cols, kinds = columns_for(doc_type), KINDS.get(doc_type)
    if not cols or not entries:
        return [], [], [], None
    statuses = [[field_status(k, e["row"].get(c, "")) for c, k in zip(cols, kinds)] for e in entries]
    scores = [sum(1 for s in st if s[0] == OK) for st in statuses]
    rows = []
    for ci, (c, k) in enumerate(zip(cols, kinds)):
        cells = [f"{statuses[ei][ci][0]} {statuses[ei][ci][1]}" for ei in range(len(entries))]
        agree = agreement(k, [e["row"].get(c, "") for e in entries])
        rows.append(cells[::-1] + [agree, c])                         # models reversed: model 1 beside field
    rows.append([f"{s}/{len(cols)} fields OK" for s in scores][::-1] + ["", "Total"])
    headers = [e["label"] for e in entries][::-1] + ["Agreement", "Field"]
    best = max(range(len(entries)), key=lambda i: (scores[i], -float(entries[i].get("time") or 0)))
    return headers, rows, scores, best

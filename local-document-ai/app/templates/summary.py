"""Key-field summary tables shown in the web lab (pure functions, no model / UI imports).

Each document type has a short list of essential fields:

    Egyptian ID card : name, birth date, address, national number (14 digits)
    Cheque           : customer name, amount, cheque date, bank name

``*_summary`` turn the structured output of the ID / cheque templates into one table
row; ``vlm_summary`` does the same for the JSON a vision-language model returned.
Everything else stays in the "raw data" view.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

DOC_GENERAL = "General"
DOC_ID = "Egyptian ID card"
DOC_CHEQUE = "Cheque"

ID_COLUMNS = ["الاسم", "تاريخ الميلاد", "العنوان", "الرقم القومي (14 رقم)"]
CHEQUE_COLUMNS = ["اسم العميل", "المبلغ", "تاريخ الشيك", "اسم البنك"]


def columns_for(doc_type: str) -> Optional[List[str]]:
    return {DOC_ID: ID_COLUMNS, DOC_CHEQUE: CHEQUE_COLUMNS}.get(doc_type)


# ------------------------------------------------------------------ helpers
def format_amount(value: Optional[float], currency: str = "") -> str:
    """12345678.0 -> '12,345,678'; 1500.5 -> '1,500.50'; None -> ''."""
    if value is None:
        return ""
    text = f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return f"{text} {currency}".strip()


def _digits(text: Any) -> str:
    """Keep only digits, mapping Arabic-Indic / Persian digits to 0-9."""
    table = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
    table.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
    return re.sub(r"\D", "", str(text or "").translate(table))


def _pick(data: Dict[str, Any], *keys: str) -> str:
    """First non-empty value among ``keys`` (case/space/underscore-insensitive), as text."""
    norm = {re.sub(r"[\s_\-]", "", str(k).lower()): v for k, v in data.items()}
    for k in keys:
        v = norm.get(re.sub(r"[\s_\-]", "", k.lower()))
        if v not in (None, "", [], {}):
            return v if isinstance(v, str) else str(v)
    return ""


def _as_row(columns: List[str], values: Iterable[str]) -> Dict[str, str]:
    return dict(zip(columns, values))


# ------------------------------------------------------------ OCR templates
def id_summary(fields: dict) -> Tuple[Dict[str, str], List[str]]:
    """One table row + warnings from the Egyptian-ID template output."""
    nid = fields.get("national_id") or {}
    number = nid.get("value", "")
    warnings: List[str] = []
    if not nid.get("valid"):
        warnings.append("National ID failed validation: " + ("; ".join(nid.get("errors", [])) or "unreadable"))
        shown = f"{number} ⚠️" if number else ""
    else:
        shown = number
    row = _as_row(ID_COLUMNS, [
        fields.get("full_name", ""),
        fields.get("birth_date") or nid.get("birth_date", ""),
        " - ".join(fields.get("address_lines") or []),
        shown,
    ])
    return row, warnings


def cheque_summary(fields: dict) -> Tuple[Dict[str, str], List[str]]:
    """One table row + warnings from the cheque template output."""
    amount = format_amount(fields.get("amount_value"), fields.get("currency", "")) or fields.get("amount_digits", "")
    row = _as_row(CHEQUE_COLUMNS, [
        fields.get("payee", ""),
        amount,
        fields.get("date") or fields.get("date_as_printed", ""),
        fields.get("bank_name_ar") or fields.get("bank_name_en", ""),
    ])
    return row, list(fields.get("issues") or [])


def template_summary(doc_type: str, fields: dict) -> Optional[Tuple[Dict[str, str], List[str]]]:
    if doc_type == DOC_ID:
        return id_summary(fields)
    if doc_type == DOC_CHEQUE:
        return cheque_summary(fields)
    return None


# ----------------------------------------------------------------------- VLM
def vlm_prompt(doc_type: str) -> str:
    """Instruction for the vision-language model, asking for exactly the fields we show."""
    base = ("Return ONLY a valid JSON object. Read values exactly as printed; if a value is not "
            "visible or you are unsure, use an empty string - never guess or invent. ")
    if doc_type == DOC_ID:
        return ("This is an Egyptian national ID card. " + base +
                'Keys: "text" (all text on the card), "name" (full name in Arabic), '
                '"birth_date" (YYYY-MM-DD), "address" (full address in Arabic), '
                '"national_id" (the 14-digit national number, digits only), '
                '"confidence" (0-100).')
    if doc_type == DOC_CHEQUE:
        return ("This is a bank cheque. " + base +
                'Keys: "text" (all text on the cheque), "customer_name" (the payee / person or '
                'company the cheque is payable to), "amount" (the amount in digits with the '
                'currency), "cheque_date" (YYYY-MM-DD), "bank_name", "confidence" (0-100).')
    return ("Extract all text and data from this image. Return as valid JSON with keys: "
            '"text" (full text), "fields" (key-value pairs found), "tables" (any tabular data), '
            '"language" (detected language), "confidence" (your confidence 0-100), '
            '"notes" (any observations).')


def vlm_summary(doc_type: str, page: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """One table row + warnings from the JSON a VLM returned for one page."""
    data = dict(page)
    if isinstance(data.get("fields"), dict):               # tolerate values nested under "fields"
        data = {**data["fields"], **data}
    warnings: List[str] = []
    if doc_type == DOC_ID:
        number = _digits(_pick(data, "national_id", "national_number", "id_number", "الرقم القومي"))
        if len(number) != 14:
            warnings.append(f"The extracted national ID is not 14 digits ({len(number)})")
        row = _as_row(ID_COLUMNS, [
            _pick(data, "name", "full_name", "الاسم"),
            _pick(data, "birth_date", "date_of_birth", "dob", "تاريخ الميلاد"),
            _pick(data, "address", "العنوان"),
            number,
        ])
        return row, warnings
    if doc_type == DOC_CHEQUE:
        row = _as_row(CHEQUE_COLUMNS, [
            _pick(data, "customer_name", "payee", "beneficiary", "client_name", "pay_to", "name"),
            _pick(data, "amount", "amount_digits", "amount_numeric"),
            _pick(data, "cheque_date", "date", "check_date"),
            _pick(data, "bank_name", "bank"),
        ])
        return row, warnings
    return {}, []


# ------------------------------------------------------------- raw data view
def flatten_fields(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Nested dict/list -> flat ``(field, value)`` rows for the raw-data table."""
    rows: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("_"):
                continue
            rows += flatten_fields(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        if obj and all(not isinstance(x, (dict, list, tuple)) for x in obj):
            rows.append((prefix, " | ".join(str(x) for x in obj)))
        else:
            for i, x in enumerate(obj, start=1):
                rows += flatten_fields(x, f"{prefix}[{i}]")
    else:
        rows.append((prefix, "" if obj is None else str(obj)))
    return rows


# ------------------------------------------------------------ vertical layout
FIELD_COL = "الحقل"
VALUE_COL = "القيمة"


def vertical_table(columns: List[str], rows: List[Dict[str, str]]) -> Tuple[List[str], List[List[str]]]:
    """Field-per-row layout for right-to-left reading.

    The FIELD column sits on the right, so reading right to left gives "field - value".
    One document  -> headers ``[القيمة, الحقل]``.
    N documents   -> ``[مستند N, ..., مستند 1, الحقل]`` (document 1 is next to its field).
    Returns ``(headers_left_to_right, rows)``."""
    if len(rows) <= 1:
        headers = [VALUE_COL, FIELD_COL]
    else:
        headers = [f"مستند {i}" for i in range(len(rows), 0, -1)] + [FIELD_COL]
    out: List[List[str]] = []
    for col in columns:
        values = [r.get(col, "") for r in rows] or [""]
        out.append((values[::-1] if len(rows) > 1 else values) + [col])
    return headers, out


def pairs_table(pairs: List[Tuple[str, str]]) -> Tuple[List[str], List[List[str]]]:
    """``(field, value)`` pairs -> same layout: ``[القيمة, الحقل]``."""
    return [VALUE_COL, FIELD_COL], [[value, field] for field, value in pairs]


# ------------------------------------------------------------------ HTML table
def html_table(headers_ltr: List[str], rows_ltr: List[List[str]], max_height: int = 420,
                highlight_headers: Any = ()) -> str:
    """Render a table whose cells are all right-aligned, columns running right to left.

    Takes the same left-to-right layout as ``vertical_table`` (field column last) and
    returns safe HTML: every value is escaped, because model output is untrusted text.
    Each cell uses ``dir="auto"`` so Arabic reads right-to-left while numbers, dates and
    Latin text keep their natural left-to-right order - all of them pinned to the right edge.
    """
    import html as _html

    def cell(text: Any) -> str:
        t = _html.escape(str(text if text is not None else "")).replace(chr(10), "<br>")
        return (t.replace(chr(0xE000), '<mark style="background:#fde68a;color:#78350f;padding:0 2px;border-radius:3px">')
                 .replace(chr(0xE001), "</mark>"))

    headers = list(reversed(headers_ltr))
    body = [list(reversed(r)) for r in rows_ltr]
    th_style = ("background:#1a1a6e;color:#ffffff;padding:8px 12px;text-align:right;"
                "border:1px solid #d1d5db;font-weight:600")
    green = set(highlight_headers or ())
    head = "".join(f'<th style="{th_style}">{cell(h)}</th>' for h in headers)
    rows_html = []
    for r in body:
        tds = []
        for i, v in enumerate(r):
            extra = "background:#f3f4f6;font-weight:600;" if i == 0 else ""
            if headers[i] in green:
                extra = "background:#dcfce7;color:#14532d;font-weight:600;"
            tds.append(f'<td dir="auto" style="{extra}padding:8px 12px;text-align:right;'
                       f'border:1px solid #e5e7eb;color:#1a1a2e;vertical-align:top">{cell(v)}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div style="max-height:{max_height}px;overflow:auto;margin:4px 0 12px 0">'
            f'<table dir="rtl" style="width:100%;border-collapse:collapse;background:#ffffff;font-size:14px">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>")

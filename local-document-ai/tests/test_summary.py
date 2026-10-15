"""Summary-table logic for the web lab (pure functions)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.templates.summary import (CHEQUE_COLUMNS, ID_COLUMNS, DOC_CHEQUE, DOC_ID, cheque_summary,
                                   flatten_fields, format_amount, html_table, id_summary, pairs_table, vertical_table, vlm_prompt, vlm_summary)

ID_FIELDS = {
    "full_name": "سارة علي حسن", "birth_date": "1999-01-01",
    "address_lines": ["شارع النصر ١٢", "المنتزه - الاسكندرية"],
    "national_id": {"value": "29901010112345", "valid": True, "errors": []},
}
CHEQUE_FIELDS = {
    "payee": "Jane Doe", "amount_value": 12345678.0, "currency": "EGP", "amount_digits": "12,345,678.000",
    "date": "2012-09-25", "date_as_printed": "25-09-2012", "bank_name_ar": "", "bank_name_en": "some bank",
    "issues": [],
}


def test_id_row_has_exactly_the_four_requested_columns():
    row, warnings = id_summary(ID_FIELDS)
    assert list(row) == ID_COLUMNS
    assert row["الرقم القومي (14 رقم)"] == "29901010112345" and not warnings
    assert row["العنوان"] == "شارع النصر ١٢ - المنتزه - الاسكندرية"


def test_invalid_national_id_is_flagged():
    bad = {**ID_FIELDS, "national_id": {"value": "2990101", "valid": False, "errors": ["expected 14 digits, got 7"]}}
    row, warnings = id_summary(bad)
    assert "⚠️" in row["الرقم القومي (14 رقم)"] and warnings


def test_cheque_row_and_amount_format():
    row, warnings = cheque_summary(CHEQUE_FIELDS)
    assert list(row) == CHEQUE_COLUMNS
    assert row["المبلغ"] == "12,345,678 EGP" and row["اسم البنك"] == "some bank"
    assert format_amount(1500.5, "EGP") == "1,500.50 EGP" and format_amount(None) == ""


def test_vlm_id_extracts_and_checks_digit_count():
    row, warnings = vlm_summary(DOC_ID, {"name": "X", "birth_date": "1999-01-01", "address": "Y",
                                         "national_id": "٢٩٩٠١٠١٠١١٢٣٤٥"})
    assert row["الرقم القومي (14 رقم)"] == "29901010112345"
    assert warnings == []                                          # no generic "unverified" note any more
    _, warnings = vlm_summary(DOC_ID, {"national_id": "123"})
    assert any("14" in w for w in warnings)


def test_vlm_cheque_accepts_common_key_names():
    row, _ = vlm_summary(DOC_CHEQUE, {"payee": "A B", "amount": "1,500,000 EGP", "date": "2022-01-30", "bank": "QNB"})
    assert row["اسم العميل"] == "A B" and row["اسم البنك"] == "QNB" and row["تاريخ الشيك"] == "2022-01-30"


def test_prompts_ask_for_the_displayed_fields():
    assert "national_id" in vlm_prompt(DOC_ID) and "birth_date" in vlm_prompt(DOC_ID)
    assert "customer_name" in vlm_prompt(DOC_CHEQUE) and "JSON" in vlm_prompt(DOC_CHEQUE)


def test_flatten_fields():
    rows = dict(flatten_fields({"a": 1, "n": {"x": "y"}, "l": ["p", "q"], "_hidden": 1, "e": None}))
    assert rows == {"a": "1", "n.x": "y", "l": "p | q", "e": ""}


def test_vertical_layout_puts_the_field_on_the_right():
    headers, rows = vertical_table(ID_COLUMNS, [id_summary(ID_FIELDS)[0]])
    assert headers == ["القيمة", "الحقل"]                       # field = last (right-most) column
    assert rows[0] == ["سارة علي حسن", "الاسم"] and len(rows) == 4
    assert rows[3] == ["29901010112345", "الرقم القومي (14 رقم)"]


def test_vertical_layout_with_several_documents():
    a, b = {"الاسم": "A"}, {"الاسم": "B"}
    headers, rows = vertical_table(["الاسم"], [a, b])
    assert headers == ["مستند 2", "مستند 1", "الحقل"]
    assert rows == [["B", "A", "الاسم"]]                        # document 1 sits next to the field


def test_pairs_table():
    assert pairs_table([("k", "v")]) == (["القيمة", "الحقل"], [["v", "k"]])


def test_html_table_aligns_every_value_right_and_escapes():
    headers, rows = vertical_table(["الاسم", "التاريخ"], [{"الاسم": "<script>alert(1)</script>", "التاريخ": "2003-01-09"}])
    out = html_table(headers, rows)
    assert "<script>" not in out and "&lt;script&gt;" in out            # untrusted model text is escaped
    assert out.count("text-align:right") >= 6                          # header + every cell
    assert 'dir="rtl"' in out and 'dir="auto"' in out
    # logical (right-to-left) column order: the field comes first, i.e. at the right edge
    assert out.index("الحقل") < out.index("القيمة")
    assert "text-align:left" not in out


if __name__ == "__main__":
    test_id_row_has_exactly_the_four_requested_columns(); test_invalid_national_id_is_flagged()
    test_cheque_row_and_amount_format(); test_vlm_id_extracts_and_checks_digit_count()
    test_vlm_cheque_accepts_common_key_names(); test_prompts_ask_for_the_displayed_fields(); test_flatten_fields()
    test_vertical_layout_puts_the_field_on_the_right(); test_vertical_layout_with_several_documents(); test_pairs_table()
    test_html_table_aligns_every_value_right_and_escapes()
    print("summary tests: OK")

"""Status-based model comparison (pure functions)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.templates.compare import agreement, compare, field_status
from app.templates.summary import DOC_CHEQUE, DOC_ID, ID_COLUMNS

GOOD = {"الاسم": "سارة علي حسن", "تاريخ الميلاد": "1999-01-01", "العنوان": "شارع النصر",
        "الرقم القومي (14 رقم)": "29901010112345"}


def test_statuses():
    assert field_status("nid", "29901010112345")[0] == "✅"
    assert field_status("nid", "2990101")[0] == "⚠️" and "7 digits" in field_status("nid", "2990101")[1]
    assert field_status("nid", "")[0] == "❌"
    assert field_status("nid", "٢٩٩٠١٠١٠١١٢٣٤٥")[0] == "✅"                 # Arabic-Indic digits
    assert field_status("nid", "99999999999999")[0] == "⚠️"                  # right length, invalid structure
    assert field_status("date", "1999-13-45")[0] == "⚠️" and field_status("date", "1999-01-01")[0] == "✅"
    assert field_status("name", "علي 123")[0] == "⚠️"


def test_agreement_hides_values_and_counts():
    assert agreement("name", ["احمد", "أحمد"]) == "🤝 All agree"            # hamza variants are equal
    assert agreement("nid", ["29901010112345", "29901010112346", "29901010112345"]).endswith("2 of 3 agree")
    assert agreement("nid", ["29901010112345", ""]) == "—"


def test_compare_table_and_best():
    bad = {**GOOD, "الرقم القومي (14 رقم)": "2990101", "العنوان": ""}
    headers, rows, scores, best = compare(DOC_ID, [
        {"label": "A", "row": GOOD, "time": 5}, {"label": "B", "row": bad, "time": 1}])
    assert scores == [4, 2] and best == 0
    assert headers[-1] == "Field" and headers[-2] == "Agreement" and headers[:2] == ["B", "A"]
    nid_row = rows[3]
    assert nid_row[-1] == ID_COLUMNS[3] and "29901010112345" not in " ".join(nid_row)     # statuses only
    assert rows[-1][-1] == "Total" and "4/4" in rows[-1][1]


def test_tie_prefers_faster_and_unknown_doc():
    _, _, _, best = compare(DOC_ID, [{"label": "slow", "row": GOOD, "time": 9}, {"label": "fast", "row": GOOD, "time": 2}])
    assert best == 1
    assert compare("General", [{"label": "x", "row": {}}]) == ([], [], [], None)


if __name__ == "__main__":
    test_statuses(); test_agreement_hides_values_and_counts(); test_compare_table_and_best(); test_tie_prefers_faster_and_unknown_doc()
    print("compare tests: OK")

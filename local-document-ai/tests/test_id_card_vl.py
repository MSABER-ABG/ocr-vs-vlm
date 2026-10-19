"""ID card through PaddleOCR-VL: number choice, text cleaning, fast number reader (no model needed)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.templates.id_card_vl import build_id_fields, pick_number
from app.templates.egypt_id import validate_national_id
from app.templates.summary import id_summary
from app.templates.text_fields import id_fields_from_text

TEXT = """جمهورية الصلاية بين
بطاقة تحقيق الشخصية
1997/.
GU8641199
مؤمن
ابراهیم محمد ابراهیم
ش. ه حياة السراياسموحه
سيدى جابر - الاسكندرية
Y 94.1.1.4 149 04"""
GOOD = validate_national_id("29301010217953")


def test_glyph_number_replaces_garbled_vl_number():
    f = build_id_fields(TEXT, GOOD)
    assert f["national_id"]["value"] == "29301010217953" and f["national_id"]["source"] == "glyph classifier"
    assert f["birth_date"] == "1993-01-01" and f["serial"] == "GU8641199"
    assert f["full_name"] == "مؤمن ابراهيم محمد ابراهيم"          # header skipped, 2-line name joined
    assert f["address_lines"] == ["ش. ه حياة السراياسموحه", "سيدي جابر - الاسكندرية"]
    assert not f["needs_review"]
    row, warnings = id_summary(f)
    assert row["الرقم القومي (14 رقم)"] == "29301010217953" and not warnings


def test_unreadable_number_is_flagged_not_invented():
    f = build_id_fields(TEXT, validate_national_id("?00747425077".replace("?", "")))
    assert not f["national_id"]["valid"] and f["needs_review"]
    row, warnings = id_summary(f)
    assert "⚠️" in row["الرقم القومي (14 رقم)"] and warnings


def test_valid_vl_number_used_when_classifier_fails_and_conflict_flagged():
    text = TEXT.replace("Y 94.1.1.4 149 04", "٢٩٣٠١٠١٠٢١٧٩٥٣")
    f = build_id_fields(text, validate_national_id("1234"))
    assert f["national_id"]["value"] == "29301010217953" and f["national_id"]["source"] == "PaddleOCR-VL text"
    other = validate_national_id("30301091501281")
    both = pick_number(GOOD, other)
    assert both["value"] == GOOD["value"] and both["conflict"] == "30301091501281"


def test_address_with_digits_and_header_variants():
    f = id_fields_from_text("بطاقة تحقيق الشخصية\nاحمد\nمحمد علي\nش ١٢٣ المنتزه\nالاسكندرية\nAB1234567")
    assert f["full_name"] == "احمد محمد علي" and f["address_lines"] == ["ش ١٢٣ المنتزه", "الاسكندرية"]
    assert f["serial"] == "AB1234567"


def test_fast_number_reader_loads_and_handles_blank_card():
    from PIL import Image
    from app.templates.id_number_reader import read_number_fast
    r = read_number_fast(Image.new("RGB", (800, 504), "white"))
    assert not r["valid"] and r["confidence"] == "low" and r["source"] == "glyph classifier"


if __name__ == "__main__":
    test_glyph_number_replaces_garbled_vl_number(); test_unreadable_number_is_flagged_not_invented()
    test_valid_vl_number_used_when_classifier_fails_and_conflict_flagged(); test_address_with_digits_and_header_variants()
    test_fast_number_reader_loads_and_handles_blank_card()
    print("id_card_vl tests: OK")

"""Cheque field extraction on synthetic OCR boxes (no models needed).

The boxes copy the layout/text of a real bilingual cheque (see README); the values
are made up, so the test contains no real customer data.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.ocr.base import OCRResult
from app.templates.cheque import (extract_fields, parse_amount, parse_date, snap, words_to_number,
                                  arabic_words_to_number, ARABIC_LABELS)


def B(text, x0, y0, x1, y1, c=0.95):
    return OCRResult(page=1, text=text, bbox=[x0, y0, x1, y1], confidence=c)


def sample_boxes(words="**Twelve Million Three Hundred And Forty Five Thousand Six Hundred And Seventy Eight Only**",
                 digits="**12,345,678.000**", date="25-09-2012"):
    return [
        B("Date", 526, 27, 554, 42), B(date, 550, 18, 637, 33),
        B("arab african international bank", 285, 46, 468, 60, 0.98),
        B("Pay by this check", 16, 94, 102, 110), B("To order", 103, 90, 148, 105),
        B("**Jane Doe", 148, 78, 230, 105), B("**", 224, 82, 241, 95), B("xXXXXXXx", 106, 108, 179, 122, 0.8),
        B("The sum of", 14, 126, 76, 143), B(words, 77, 122, 598, 138),
        B("Name", 14, 171, 50, 187), B("اسم الساحب", 279, 172, 338, 191),
        B("EGP", 439, 170, 472, 187), B(digits, 499, 168, 617, 186),
        B("Signature", 15, 216, 65, 230),
    ]


def test_words_to_number():
    assert words_to_number("Twelve Million Three Hundred And Forty Five Thousand Six Hundred And Seventy Eight Only")[0] == 12_345_678
    assert words_to_number("One Hundred Only")[0] == 100
    assert words_to_number("Five Thousand")[0] == 5000
    assert words_to_number("Fourty Five")[0] == 45                      # OCR typo repaired
    assert words_to_number("hello")[0] is None
    assert words_to_number("Twelve Banana Thousand")[1] == 1            # unknown word is counted


def test_arabic_words_to_number():
    cases = {
        "فقط مليون وخمسمائة ألف جنيه مصري لا غير": 1_500_000,
        "فقط ثمانية الاف وخمسمائة وخمسة وستون جنيها مصريا لا غير": 8_565,
        "ألفان وثلاثمائة": 2_300,
        "مائة وخمسة وعشرون": 125,
        "احد عشر ألف": 11_000,
        "ثلاثة ملايين": 3_000_000,
    }
    for text, want in cases.items():
        assert arabic_words_to_number(text)[0] == want, (text, arabic_words_to_number(text))


def test_arabic_words_tolerate_ocr_errors():
    # real OCR misreads: ثمائية / خسة / مصريالا
    got, unknown = arabic_words_to_number("فقط ثمائية الاف وخمسمائة وخسة وستون جنيها مصريالا غير")
    assert got == 8_565 and unknown == 0, (got, unknown)


def test_date_and_amount_parsing():
    assert parse_date("25-09-2012") == "2012-09-25"
    assert parse_date("25 / 9 / 2012") == "2012-09-25"
    assert parse_date("31-02-2012") is None
    assert parse_amount("**12,345,678.000**") == 12345678.0
    assert parse_amount("25") is None                                   # too few digits to be an amount


def test_snap_only_printed_labels():
    assert snap("ميلناً وقدرء", ARABIC_LABELS, 0.7) == ("مبلغاً وقدره", True)
    assert snap("الثاريع", ARABIC_LABELS, 0.7) == ("التاريخ", True)
    assert snap("محمد أحمد", ARABIC_LABELS, 0.7)[1] is False             # a person's name is left alone
    assert snap("صابر", ARABIC_LABELS, 0.7)[1] is False


def test_extracts_all_fields_and_cross_checks():
    f, issues = extract_fields(sample_boxes(), 333)
    assert f["date"] == "2012-09-25"
    assert f["payee"] == "Jane Doe"
    assert f["payee_blank_line"] == "xXXXXXXx"
    assert f["currency"] == "EGP" and f["amount_value"] == 12345678.0
    assert f["amount_words_value"] == 12_345_678 and f["amount_words_match_digits"] is True
    assert f["signatory_name"] == "" and f["bank_name_en"] == "arab african international bank"
    assert not issues and not f["needs_review"]


def test_mismatch_is_flagged_not_hidden():
    f, issues = extract_fields(sample_boxes(digits="**12,345,679.000**"), 333)
    assert f["amount_words_match_digits"] is False and f["needs_review"]
    assert any("mismatch" in i for i in issues)


def test_bad_date_is_flagged():
    f, issues = extract_fields(sample_boxes(date="31-02-2012"), 333)
    assert f["date"] == "" and f["needs_review"]





def arabic_cheque_boxes():
    """Layout of a QNB-style Arabic cheque: labels `PAY TO`/`AGAINST THIS CHEQUE`, Arabic payee + words."""
    return [
        B("QNB ALAHLI", 50, 40, 150, 60), B("بنك قطر الوطني الأهلي", 270, 45, 400, 62),
        B("DATE", 410, 62, 440, 76), B("30/01/2022", 470, 60, 560, 76),
        B("AGAINST THIS CHEQUE", 30, 100, 140, 112), B("PAY TO", 30, 114, 70, 126),
        B("احمد محمد محمود احمد", 270, 108, 420, 126), B("إدفعوا بموجب", 560, 95, 620, 110),
        B("هذا الشيك", 575, 110, 625, 125),
        B("THE SUM OF", 30, 155, 95, 167),
        B("فقط مليون وخمسمائة الف جنيه مصري لا غير", 290, 152, 440, 170), B("مبلغاً وقدره", 445, 155, 505, 168),
        B("EGP", 470, 185, 500, 200), B("#1,500,000#", 520, 183, 600, 200),
        B("SIGNATURE", 280, 262, 340, 274), B("صرف للمستفيد الأول", 5, 10, 60, 60),
    ]


def test_arabic_cheque_payee_and_words():
    f, issues = extract_fields(arabic_cheque_boxes(), 327)
    assert f["date"] == "2022-01-30"
    assert f["payee"] == "احمد محمد محمود احمد", f["payee"]
    assert f["amount_value"] == 1_500_000 and f["amount_words_value"] == 1_500_000
    assert f["amount_words_match_digits"] is True and not issues, issues


def test_multiline_english_words_are_joined():
    boxes = sample_boxes(words="**Twelve Million Three Hundred And Forty Five")
    boxes.append(B("Thousand Six Hundred And Seventy Eight Only**", 28, 150, 420, 168))
    f, issues = extract_fields(boxes, 344)
    assert f["amount_words_value"] == 12_345_678 and f["amount_words_match_digits"] is True, f["amount_words"]


if __name__ == "__main__":
    test_words_to_number(); test_date_and_amount_parsing(); test_snap_only_printed_labels()
    test_extracts_all_fields_and_cross_checks(); test_mismatch_is_flagged_not_hidden(); test_bad_date_is_flagged()
    test_arabic_words_to_number(); test_arabic_words_tolerate_ocr_errors()
    test_arabic_cheque_payee_and_words(); test_multiline_english_words_are_joined()
    print("cheque tests: OK")

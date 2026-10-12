"""visual_to_logical: CTC (visual-order) text -> logical reading order."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from app.ocr.text import visual_to_logical


def test_words_and_letters():
    assert visual_to_logical("ةقاطب") == "بطاقة"
    assert visual_to_logical("يموق مقر ةقاطب") == "بطاقة رقم قومي"        # words flip too


def test_numbers_keep_their_order():
    assert visual_to_logical("ةايح ٥٠ ش") == "ش ٥٠ حياة"                  # Arabic-Indic digits
    assert visual_to_logical("ةايح 50 ش") == "ش 50 حياة"                  # Western digits
    assert visual_to_logical("٢٠٠٣/٠١/٠٩") == "٢٠٠٣/٠١/٠٩"               # no Arabic letters: untouched
    assert visual_to_logical("ةنس 2003/01/09") == "2003/01/09 سنة"


def test_latin_and_codes_untouched():
    assert visual_to_logical("AB1234567") == "AB1234567"
    assert visual_to_logical("Roy Ang") == "Roy Ang"
    assert visual_to_logical("ةقاطب JB257") == "JB257 بطاقة"


if __name__ == "__main__":
    test_words_and_letters(); test_numbers_keep_their_order(); test_latin_and_codes_untouched()
    print("text_order tests: OK")

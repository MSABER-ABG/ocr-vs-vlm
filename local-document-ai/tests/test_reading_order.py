"""Reading-order unit tests (no models needed)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.layout.reading_order import sort_reading_order
from app.ocr.base import OCRResult


def box(text, x, y=100, w=80, h=30):
    return OCRResult(page=1, text=text, bbox=[x, y, x + w, y + h], confidence=0.9)


def test_english_sentence_next_to_arabic_label():
    # Cheque row: English sentence on the left, Arabic label on the right.
    words = [box("The", 10), box("sum", 100), box("of", 190), box("Twelve", 280),
             box("مبلغا", 600), box("وقدره", 520)]
    line = sort_reading_order(words)[0]
    assert line.text == "The sum of Twelve مبلغا وقدره", line.text
    assert line.direction == "ltr"


def test_pure_arabic_line_is_right_to_left():
    words = [box("قومي", 10), box("رقم", 120), box("بطاقة", 220)]       # rightmost = first
    line = sort_reading_order(words)[0]
    assert line.text == "بطاقة رقم قومي", line.text
    assert line.direction == "rtl"


def test_numbers_stay_left_to_right():
    words = [box("288", 10), box("01123", 150), box("00553", 300)]
    assert sort_reading_order(words)[0].text == "288 01123 00553"


def test_digits_inside_arabic_line_follow_the_arabic_run():
    words = [box("حياة", 10), box("٥٠", 120), box("ش", 200)]            # printed right-to-left
    assert sort_reading_order(words)[0].text == "ش ٥٠ حياة"


def test_lines_are_top_to_bottom():
    lines = sort_reading_order([box("second", 10, y=300), box("first", 10, y=50)])
    assert [l.text for l in lines] == ["first", "second"]


if __name__ == "__main__":
    test_english_sentence_next_to_arabic_label()
    test_pure_arabic_line_is_right_to_left()
    test_numbers_stay_left_to_right()
    test_digits_inside_arabic_line_follow_the_arabic_run()
    test_lines_are_top_to_bottom()
    print("reading_order tests: OK")

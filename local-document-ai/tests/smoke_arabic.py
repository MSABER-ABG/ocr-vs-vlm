"""Smoke test: Arabic must come out in logical order, numbers left-to-right.

Run from the project root:  python -m tests.smoke_arabic
Extra dev deps for the synthetic image: pip install arabic-reshaper python-bidi
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.ocr.paddle_v4 import to_logical_order


def test_to_logical_order():
    assert to_logical_order("ةقاطب") == "بطاقة"
    assert to_logical_order("دمحأ") == "أحمد"
    assert to_logical_order("ةيب رعلا") == "العر بية"        # multi-word box: words flip too
    assert to_logical_order("00553") == "00553"              # no Arabic -> untouched
    # visual "ةقاطب 123" = number sits to the RIGHT of the word -> read first; digits stay 123, not 321
    assert to_logical_order("ةقاطب 123") == "123 بطاقة"
    print("to_logical_order: OK")


def test_pipeline_on_synthetic_image():
    import arabic_reshaper
    from bidi.algorithm import get_display
    from PIL import Image, ImageDraw, ImageFont

    from app.config import Config
    from app.output import to_text
    from app.pipeline import DocumentAI

    phrases = ["بطاقة رقم قومي", "محمد أحمد", "مصطفى صابر"]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56)
    img = Image.new("RGB", (1300, 130 * len(phrases) + 40), "white")
    d = ImageDraw.Draw(img)
    for i, w in enumerate(phrases):
        d.text((60, 30 + i * 130), get_display(arabic_reshaper.reshape(w)), fill="black", font=font)

    text = to_text(DocumentAI(Config(languages=["ar"])).process_pages([img]))
    print(text)
    for word in ["بطاقة", "أحمد", "محمد", "مصطفى", "صابر"]:
        assert word in text, f"missing {word!r} in output"
    print("pipeline Arabic order: OK")


if __name__ == "__main__":
    test_to_logical_order()
    test_pipeline_on_synthetic_image()

"""Smoke test for the ONNX PP-OCRv5 engine (downloads ~100 MB of models on first run).

    python -m tests.smoke_onnx          (needs: pip install arabic-reshaper python-bidi)
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    import arabic_reshaper
    from bidi.algorithm import get_display
    from PIL import Image, ImageDraw, ImageFont

    from app.config import Config
    from app.output import to_text
    from app.pipeline import DocumentAI

    phrases = ["بطاقة رقم قومي", "محمد أحمد", "مصطفى صابر", "جمهورية مصر العربية"]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 56)
    img = Image.new("RGB", (1300, 130 * len(phrases) + 40), "white")
    d = ImageDraw.Draw(img)
    for i, w in enumerate(phrases):
        d.text((60, 30 + i * 130), get_display(arabic_reshaper.reshape(w)), fill="black", font=font)

    text = to_text(DocumentAI(Config(engine="onnx_v5")).process_pages([img]))
    print(text)
    for p in phrases:
        assert p in text, f"missing {p!r}"
    print("onnx_v5 engine: OK")


if __name__ == "__main__":
    main()

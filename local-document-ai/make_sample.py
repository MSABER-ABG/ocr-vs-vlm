"""Create a small synthetic sample (image + 2-page PDF) for smoke-testing."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Path("input").mkdir(exist_ok=True)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)


def page(lines):
    img = Image.new("RGB", (1200, 700), "white")
    d = ImageDraw.Draw(img)
    for i, (x, text) in enumerate(lines):
        d.text((x, 60 + i * 120), text, fill="black", font=font)
    return img


p1 = page([(80, "INVOICE No 2025-0417"), (80, "Customer: Ahmed Saber"), (80, "Total 1,250.00 EGP"), (700, "Page 1")][:3] + [(700, "Page 1")])
p2 = page([(80, "Second page text"), (80, "Phone 0100 123 4567")])
p1.save("input/sample.png")
p1.save("input/sample.pdf", save_all=True, append_images=[p2])
print("created input/sample.png and input/sample.pdf")

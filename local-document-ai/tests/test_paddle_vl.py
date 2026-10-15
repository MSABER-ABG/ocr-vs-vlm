"""PaddleOCR-VL output conversion (pure functions, no model needed)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.ocr.paddle_vl import blocks_to_text, html_to_text


def test_table_is_flattened_to_rows():
    html_table = ("<table><tr><td>DATE</td><td>30/01/2022</td></tr>"
                  "<tr><td>EGP</td><td>#1,500,000#</td></tr></table>")
    assert html_to_text(html_table) == "DATE | 30/01/2022\nEGP | #1,500,000#"


def test_literal_backslash_n_and_entities():
    assert html_to_text("<td>QNB\\nALAHLI</td>") == "QNB ALAHLI"
    assert html_to_text("a &amp; b") == "a & b"


def test_blocks_joined_across_pages():
    pages = [{"blocks": [{"label": "text", "content": "first"}, {"label": "text", "content": ""}]},
             {"blocks": [{"label": "text", "content": "second"}]}]
    assert blocks_to_text(pages) == "first\n\n--- Page Break ---\n\nsecond"


if __name__ == "__main__":
    test_table_is_flattened_to_rows(); test_literal_backslash_n_and_entities(); test_blocks_joined_across_pages()
    print("paddle_vl tests: OK")

"""Scoring against the reference model (pure functions)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.templates.compare_ref import compare_to_reference, mark_noise, value_score, words, words_score
from collections import Counter


def test_words_and_noise():
    pct, detail = words_score("مؤمن ابراهيم محمد ابراهيم", Counter(words("مؤمن ابراهيم")))
    assert round(pct) == 50 and detail == "2/4 words"
    marked, extra = mark_noise("مؤمن ابراهيم", "مؤمن ابراهيم ش ٥٠")
    assert extra == 2 and "ش" in marked


def test_digits_are_partial_not_all_or_nothing():
    assert value_score("nid", "29901011234567", "29901011234567")[0] == 100.0
    pct, detail = value_score("nid", "29901011234567", "29901011234568")
    assert detail == "13/14 digits" and 90 < pct < 100
    assert value_score("date", "2025-01-05", "05/01/2025")[0] == 100.0
    assert value_score("date", "2025-01-05", "2025-01-06")[1] == "7/8 digits"


def test_reference_column_is_first_and_scores_100():
    fields = [("name", "name", "Ahmed Ali")]
    headers, rows, avgs, best, _ = compare_to_reference(
        "General", fields, [{"label": "M", "time": 1, "row": {"name": "Ahmed"}}], "Reference")
    assert headers == ["M", "Reference", "Field"] and avgs == [50.0] and best == 0


if __name__ == "__main__":
    test_words_and_noise(); test_digits_are_partial_not_all_or_nothing(); test_reference_column_is_first_and_scores_100()
    print("compare_ref tests: OK")

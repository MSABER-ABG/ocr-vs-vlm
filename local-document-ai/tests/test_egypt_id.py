"""Unit tests for national-ID validation. Uses synthetic numbers only (no real people)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.templates.digits import to_western_digits
from app.templates.egypt_id import validate_national_id


def make(century, yymmdd, gov, seq, check="5") -> str:
    """Compose C YYMMDD GG SSSS K from its parts."""
    n = f"{century}{yymmdd}{gov}{seq}{check}"
    assert len(n) == 14, n
    return n


def test_valid_numbers():
    r = validate_national_id(make("2", "990101", "01", "1234"))          # 13th digit 4 -> even
    assert r["valid"], r
    assert r["birth_date"] == "1999-01-01"
    assert r["birth_governorate"] == "Cairo" and r["gender"] == "female"

    r = validate_national_id(make("3", "050617", "21", "1237"))          # 13th digit 7 -> odd
    assert r["valid"], r
    assert r["birth_date"] == "2005-06-17"
    assert r["birth_governorate"] == "Giza" and r["gender"] == "male"


def test_rejects_bad_numbers():
    assert not validate_national_id(make("2", "990101", "01", "1234")[:13])["valid"]   # 13 digits
    assert not validate_national_id(make("4", "990101", "01", "1234"))["valid"]        # century 4
    assert not validate_national_id(make("2", "991332", "01", "1234"))["valid"]        # month 13 / day 32
    assert not validate_national_id(make("2", "990230", "01", "1234"))["valid"]        # 30 Feb
    assert not validate_national_id(make("2", "990101", "99", "1234"))["valid"]        # governorate 99
    assert not validate_national_id(make("3", "991231", "01", "1234"))["valid"]        # born in 2099
    assert not validate_national_id("")["valid"]


def test_digit_normalisation():
    assert to_western_digits("٢٩٩٠١٠١٠١١٢٣٤٥") == "29901010112345"
    assert to_western_digits("۲۹۹") == "299"                               # Persian digits
    assert to_western_digits("A1-2 ٣") == "123"


if __name__ == "__main__":
    test_valid_numbers()
    test_rejects_bad_numbers()
    test_digit_normalisation()
    print("egypt_id tests: OK")

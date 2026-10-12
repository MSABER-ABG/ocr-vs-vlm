"""Regression check on the real cheque photos kept in input/ (skipped if absent).

Expected values were read from the images by eye. NOTE: these same images were used
while building the template, so a pass proves no regression, NOT generalisation.

    python -m tests.regress_cheques
"""
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EXPECTED = {
    "cheque.jpg":         dict(date="2012-09-25", payee="Roy Ang", amount=12_345_678),
    "cheque_qnb_en.jpg":  dict(date="2012-09-25", payee="Roy Ang", amount=12_345_678),
    "cheque_qnb_ar.jpg":  dict(date="2022-01-30", payee="احمد محمد محمود احمد", amount=1_500_000),
    "cheque_misr.jpg":    dict(date="2020-10-30", payee="محمد احمد المنصور", amount=8_565),   # substring
}


def main():
    from PIL import Image
    from app.config import Config
    from app.pipeline import DocumentAI
    from app.templates.cheque import ChequeReader

    reader = ChequeReader(DocumentAI(Config(engine="onnx_v5")))
    failed = 0
    for name, want in EXPECTED.items():
        path = Path("input") / name
        if not path.exists():
            print(f"SKIP {name} (not found)")
            continue
        f = reader.read(Image.open(path).convert("RGB")).fields
        checks = {
            "date": f["date"] == want["date"],
            "payee": want["payee"] in f["payee"],
            "amount_digits": f["amount_value"] == want["amount"],
            "amount_words": f["amount_words_value"] == want["amount"],
            "words_match_digits": f["amount_words_match_digits"] is True,
            "no_issues": not f["issues"],
        }
        bad = [k for k, ok in checks.items() if not ok]
        failed += bool(bad)
        print(f"{'OK  ' if not bad else 'FAIL'} {name}" + (f"  -> {bad}  {f['issues']}" if bad else ""))
    assert not failed, f"{failed} cheque(s) regressed"
    print("cheque regression: OK")


if __name__ == "__main__":
    main()

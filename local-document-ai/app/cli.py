"""Command line: python -m app.cli input/scan.pdf -o output --gpu"""
import argparse
import sys
import time
from pathlib import Path

from app.config import Config
from app.output import write_outputs
from app.pipeline import DocumentAI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local Document AI (no VLM, no cloud)")
    ap.add_argument("inputs", nargs="+", help="PDF / PNG / JPG files")
    ap.add_argument("-o", "--output", default="output", help="output directory")
    ap.add_argument("-l", "--lang", nargs="+", default=["ar", "en"], help="e.g. ar en")
    ap.add_argument("--gpu", action="store_true", help="use NVIDIA GPU (v4 engine)")
    ap.add_argument("--engine", choices=["paddle_v4", "onnx_v5", "paddle_v5"], default="paddle_v4",
                    help="onnx_v5 = accurate PP-OCRv5, paddle_v4 = fast PP-OCRv4")
    ap.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--template", choices=["egypt-id"],
                    help="read a fixed-layout document (Egyptian national ID card)")
    ap.add_argument("--raw-arabic-order", action="store_true",
                    help="keep the recognizer's raw (visual-order) Arabic text")
    args = ap.parse_args(argv)

    config = Config(
        engine=args.engine,
        languages=args.lang,
        use_gpu=args.gpu,
        pdf_dpi=args.dpi,
        preprocess=not args.no_preprocess,
        arabic_logical_order=not args.raw_arabic_order,
    )
    # Reconfigure stdout so Arabic prints correctly on Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    engine = DocumentAI(config)

    if args.template == "egypt-id":
        import json
        from app.io import load_document
        from app.templates.egypt_id import EgyptIDReader

        reader = EgyptIDReader(engine)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        for path in args.inputs:
            res = reader.read(load_document(path, config)[0])
            out = Path(args.output) / f"{Path(path).stem}_id.json"
            out.write_text(json.dumps(res.fields, ensure_ascii=False, indent=2), encoding="utf-8")
            nid = res.fields["national_id"]
            print(f"{Path(path).name}: national_id valid={nid['valid']} "
                  f"needs_review={res.fields['needs_review']} -> {out}")
        return 0

    for path in args.inputs:
        t0 = time.time()
        doc = engine.process(path)
        paths = write_outputs(doc, args.output)
        n_lines = sum(len(p.lines) for p in doc.pages)
        print(f"{Path(path).name}: {len(doc.pages)} page(s), {n_lines} lines, "
              f"{time.time() - t0:.1f}s -> {paths['json']}, {paths['txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

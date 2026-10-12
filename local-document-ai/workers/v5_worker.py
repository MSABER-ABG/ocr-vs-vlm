"""PP-OCRv5 worker process.

Runs inside the isolated environment that has PaddlePaddle 3.x / PaddleOCR 3.x
(``.venv_v5``). The main application talks to it over stdin/stdout, one JSON
object per line, so the two Paddle generations never share a process.

Protocol (every reply line is prefixed with ``@@``):
    -> {"op": "ocr", "path": "page.png"}
    <- @@{"ok": true, "items": [{"text": "...", "score": 0.97, "poly": [[x, y], ...]}]}
    -> {"op": "rec", "paths": ["crop1.png", ...]}
    <- @@{"ok": true, "items": [{"text": "...", "score": 0.9}, ...]}
    -> {"op": "quit"}
"""
import io
import json
import os
import sys

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Keep the protocol channel clean: anything libraries print goes to stderr.
_proto = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stdout = sys.stderr


def _reply(obj) -> None:
    _proto.write("@@" + json.dumps(obj, ensure_ascii=False) + "\n")
    _proto.flush()


def main() -> None:
    limit = int(os.environ.get("V5_DET_LIMIT", "1280"))
    threads = int(os.environ.get("V5_CPU_THREADS", "8"))

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="ar",
        ocr_version="PP-OCRv5",
        text_det_limit_type="max",
        text_det_limit_side_len=limit,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,          # oneDNN + PIR crashes on Windows CPU in Paddle 3.x
        cpu_threads=threads,
    )
    rec_model = None
    _reply({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                break
            if op == "ocr":
                r = ocr.predict(req["path"])[0]
                items = [
                    {"text": t, "score": float(s), "poly": [[float(x), float(y)] for x, y in p]}
                    for t, s, p in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])
                ]
                _reply({"ok": True, "items": items})
            elif op == "rec":
                if rec_model is None:
                    from paddleocr import TextRecognition
                    rec_model = TextRecognition(
                        model_name="arabic_PP-OCRv5_mobile_rec", enable_mkldnn=False
                    )
                out = []
                for path in req["paths"]:
                    res = rec_model.predict(input=path)[0]
                    out.append({"text": res["rec_text"], "score": float(res["rec_score"])})
                _reply({"ok": True, "items": out})
            else:
                _reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:                       # keep the worker alive
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()

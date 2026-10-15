"""PaddleOCR-VL one-shot runner (runs inside the isolated PaddlePaddle-3 environment).

    python vl_run.py <out.json> <image1> [image2 ...]

Weights are read from local folders (env ABG_MODELS_DIR, default: ../models), newest first:

    PaddleOCR-VL-1.6/  + PP-DocLayoutV3/   -> pipeline v1.6  (latest)
    PaddleOCR-VL/      (incl. PP-DocLayoutV2/) -> pipeline v1.0
    (none of them)                          -> library default (v1.6, auto-downloaded)

Loads the model, parses every image, writes the layout blocks to <out.json> and exits,
which frees the ~4.5 GB the model needs. Progress goes to stderr only.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def build_pipeline():
    from paddleocr import PaddleOCRVL

    models = Path(os.environ.get("ABG_MODELS_DIR") or Path(__file__).resolve().parents[1] / "models")

    v16, layout3 = models / "PaddleOCR-VL-1.6", models / "PP-DocLayoutV3"
    if (v16 / "model.safetensors").exists() and layout3.exists():
        _log(f"using local weights (v1.6): {v16}")
        pipe = PaddleOCRVL(pipeline_version="v1.6",
                           vl_rec_model_dir=str(v16),
                           layout_detection_model_dir=str(layout3))
        return pipe, "PaddleOCR-VL v1.6 (local weights)"

    v10 = models / "PaddleOCR-VL"
    if (v10 / "model.safetensors").exists() and (v10 / "PP-DocLayoutV2").exists():
        _log(f"using local weights (v1.0): {v10}")
        pipe = PaddleOCRVL(pipeline_version="v1",
                           vl_rec_model_dir=str(v10),
                           layout_detection_model_dir=str(v10 / "PP-DocLayoutV2"))
        return pipe, "PaddleOCR-VL v1.0 (local weights)"

    _log("no local weights found - using the library default (v1.6, auto-downloaded)")
    return PaddleOCRVL(), "PaddleOCR-VL v1.6 (auto-downloaded)"


def main() -> None:
    out_json, images = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    pipe, model_label = build_pipeline()
    load_s = time.time() - t0

    pages = []
    for path in images:
        t1 = time.time()
        blocks = []
        for res in pipe.predict(path):
            for b in (res["parsing_res_list"] if "parsing_res_list" in res else []):
                blocks.append({"label": str(b.label), "content": b.content or ""})
        pages.append({"blocks": blocks, "seconds": round(time.time() - t1, 1)})
        _log(f"parsed {path} in {time.time() - t1:.0f}s")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"pages": pages, "load_seconds": round(load_s, 1), "model": model_label},
                  f, ensure_ascii=False)


if __name__ == "__main__":
    main()

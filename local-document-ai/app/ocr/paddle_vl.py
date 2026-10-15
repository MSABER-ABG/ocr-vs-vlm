"""PaddleOCR-VL (0.9B vision-language document parser) - experimental local model.

Weights are read from local folders under ``models/`` (override with ABG_MODELS_DIR),
newest first: ``PaddleOCR-VL-1.6`` + ``PP-DocLayoutV3`` (latest), else ``PaddleOCR-VL``
(v1.0, includes its own layout model). If none exist the library falls back to its own
default (v1.6, auto-downloaded).

It needs PaddlePaddle 3.x, so it runs as a one-shot subprocess in the isolated
``.venv_v5`` environment. One process per request: the ~4.5 GB the model needs is
released as soon as the page is parsed, which keeps the web app safe on small machines.

Measured on a 16 GB laptop CPU (no GPU): ~70 s to load + ~85 s per page, ~4.5 GB RAM.
On a bilingual Arabic cheque it invented Arabic words/diacritics and skipped the amount
in words, so treat its Arabic output as unverified.
"""
import html
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "workers" / "vl_run.py"
MIN_FREE_RAM_GB = 3.0


def find_v5_python() -> str:
    """Interpreter that has PaddlePaddle 3.x + paddleocr[doc-parser]."""
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    candidates = [os.environ.get("ABG_V5_PYTHON", ""),
                  str(_ROOT / ".venv_v5" / exe),
                  str(_ROOT.parent / ".venv_v5" / exe),                              # repo root
                  str(_ROOT.parents[1] / "local-document-ai" / ".venv_v5" / exe)]   # sibling checkout
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        "PaddleOCR-VL needs the PaddlePaddle-3 environment (.venv_v5). Create it with:\n"
        "  python -m venv .venv_v5 && .venv_v5/Scripts/python -m pip install "
        "'paddleocr[doc-parser]'\n(or point ABG_V5_PYTHON at an interpreter that has it)."
    )


def html_to_text(content: str) -> str:
    """Flatten the HTML tables / markup PaddleOCR-VL emits into plain readable text."""
    text = content.replace("\\n", " ")                       # the model writes literal "\n" in cells
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]\s*>", "  |  ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    lines = []
    for ln in html.unescape(text).splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        ln = re.sub(r"(\|\s*)+$", "", ln).strip()             # trailing empty cells
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def blocks_to_text(pages: List[dict]) -> str:
    out = []
    for page in pages:
        out.append("\n".join(t for b in page["blocks"] if (t := html_to_text(b["content"]))))
    return "\n\n--- Page Break ---\n\n".join(out)


def check_memory() -> None:
    try:
        import psutil
    except ImportError:
        return
    free = psutil.virtual_memory().available / 1e9
    if free < MIN_FREE_RAM_GB:
        raise MemoryError(
            f"PaddleOCR-VL needs about 4.5 GB of RAM and only {free:.1f} GB is free. "
            "Close other programs (browser tabs, other models) and try again."
        )


def parse_images(images: List[Image.Image], timeout: int = 1800) -> dict:
    """Run PaddleOCR-VL on PIL images. Returns ``{"pages": [...], "load_seconds": float}``."""
    check_memory()
    python = find_v5_python()
    cache = _ROOT / "models"
    tmp_root = cache / "tmp"                      # keep temp files off a nearly-full system drive
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "ABG_MODELS_DIR": os.environ.get("ABG_MODELS_DIR", str(cache)),
           "PADDLE_PDX_CACHE_HOME": str(cache / "paddlex_cache"),
           "HF_HOME": str(cache / "hf"),
           "TMP": str(tmp_root), "TEMP": str(tmp_root),
           "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory(dir=tmp_root) as d:
        paths = []
        for i, img in enumerate(images):
            p = str(Path(d) / f"page{i + 1}.png")
            img.convert("RGB").save(p)
            paths.append(p)
        out_json = str(Path(d) / "result.json")
        proc = subprocess.run([python, str(_WORKER), out_json, *paths], env=env, cwd=str(_ROOT),
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        if proc.returncode != 0 or not Path(out_json).exists():
            tail = (proc.stderr or "")[-600:]
            raise RuntimeError(f"PaddleOCR-VL failed (exit {proc.returncode}): {tail}")
        return json.loads(Path(out_json).read_text(encoding="utf-8"))

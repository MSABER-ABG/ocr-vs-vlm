"""PP-OCRv5 Arabic engine (accurate) - runs in an isolated worker process.

Why a separate process: PP-OCRv5 needs PaddlePaddle 3.x, while the rest of the
project (and the Streamlit lab) uses 2.6.x. Keeping them in different
interpreters avoids any package/DLL clash and keeps this engine swappable.
"""
import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from app.config import Config
from app.ocr.base import OCREngine, OCRResult

_ROOT = Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "workers" / "v5_worker.py"
_PREFIX = "@@"


def default_v5_python() -> str:
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return str(_ROOT / ".venv_v5" / exe)


class _WorkerClient:
    """Talks JSON-lines to the worker; thread-safe, restarts on crash."""

    def __init__(self, python: str, cpu_threads: int, det_limit: int):
        if not Path(python).exists():
            raise FileNotFoundError(
                f"PP-OCRv5 environment not found: {python}\n"
                "Create it with:  python -m venv .venv_v5 && "
                ".venv_v5/Scripts/python -m pip install paddlepaddle paddleocr"
            )
        self._python, self._lock = python, threading.Lock()
        self._env = {**os.environ, "V5_CPU_THREADS": str(cpu_threads),
                     "V5_DET_LIMIT": str(det_limit), "PYTHONIOENCODING": "utf-8"}
        self._proc = None
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        log = open(_ROOT / ".v5_worker.log", "ab")
        self._proc = subprocess.Popen(
            [self._python, "-u", str(_WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            text=True, encoding="utf-8", bufsize=1, cwd=str(_ROOT), env=self._env,
        )
        first = self._read()
        if not first.get("ready"):
            raise RuntimeError(f"PP-OCRv5 worker failed to start: {first}")

    def _read(self) -> dict:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("PP-OCRv5 worker exited unexpectedly (see .v5_worker.log)")
            if line.startswith(_PREFIX):
                return json.loads(line[len(_PREFIX):])

    def call(self, request: dict) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                self._start()
            self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "unknown worker error"))
        return reply

    def close(self) -> None:
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.stdin.write('{"op": "quit"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
        except Exception:
            if self._proc:
                self._proc.kill()


class _RecShim:
    """Mimics ``PaddleOCR.ocr(img, det=False)`` so digit reading works unchanged."""

    def __init__(self, client: _WorkerClient):
        self._client = client

    def ocr(self, img, det=False, cls=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "crop.png")
            Image.fromarray(np.asarray(img)).save(path)
            item = self._client.call({"op": "rec", "paths": [path]})["items"][0]
        return [[(item["text"], item["score"])]]


class PaddleV5Engine(OCREngine):
    def __init__(self, config: Config):
        self.config = config
        self._client = _WorkerClient(
            config.v5_python or default_v5_python(), config.v5_cpu_threads, config.v5_det_limit
        )

    def get_reader(self, lang: str = "ar"):
        return _RecShim(self._client)

    def predict(self, image: Image.Image) -> List[OCRResult]:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "page.png")
            image.convert("RGB").save(path)
            items = self._client.call({"op": "ocr", "path": path})["items"]
        results: List[OCRResult] = []
        for it in items:
            text = it["text"].strip()
            if not text or it["score"] < self.config.min_confidence:
                continue
            xs = [p[0] for p in it["poly"]]
            ys = [p[1] for p in it["poly"]]
            # v5 returns Arabic in logical order already: no reversal needed.
            results.append(OCRResult(page=0, text=text,
                                     bbox=[min(xs), min(ys), max(xs), max(ys)],
                                     confidence=float(it["score"])))
        return results

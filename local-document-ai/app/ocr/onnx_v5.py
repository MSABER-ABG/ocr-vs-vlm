"""PP-OCRv5 Arabic engine running on onnxruntime (CPU) - no PaddlePaddle needed.

Detector : PP-OCRv5_server_det  (DB text detector)
Recognizer: arabic_PP-OCRv5_mobile_rec (CTC)

Models are the official PaddlePaddle ONNX exports from Hugging Face and are
downloaded once into ``models/``. Pre/post-processing mirrors PaddleOCR's own
(DBPostProcess, CTCLabelDecode) so results match the Paddle 3 pipeline.
"""
import math
import urllib.request
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from app.config import Config
from app.ocr.base import OCREngine, OCRResult
from app.ocr.text import visual_to_logical

_ROOT = Path(__file__).resolve().parents[2]
_HF = "https://huggingface.co/PaddlePaddle/{repo}/resolve/main/{file}"
MODELS = {
    "det": ("PP-OCRv5_server_det_onnx", "ppocrv5_server_det"),
    "rec": ("arabic_PP-OCRv5_mobile_rec_onnx", "ppocrv5_arabic_rec"),
}


def ensure_models(models_dir: Path) -> dict:
    """Download the ONNX models once; return their directories."""
    dirs = {}
    for key, (repo, folder) in MODELS.items():
        d = Path(models_dir) / folder
        d.mkdir(parents=True, exist_ok=True)
        for fname in ("inference.onnx", "inference.yml"):
            target = d / fname
            if not target.exists():
                tmp = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(_HF.format(repo=repo, file=fname), tmp)
                tmp.replace(target)
        dirs[key] = d
    return dirs


# ------------------------------------------------------------------ detection
def _mini_box(contour) -> Tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    box = np.array([pts[i1], pts[i2], pts[i3], pts[i4]], dtype=np.float32)   # tl, tr, br, bl
    return box, min(rect[1])


def _box_score(prob: np.ndarray, box: np.ndarray) -> float:
    h, w = prob.shape
    box = box.copy()
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0])


def _unclip(box: np.ndarray, ratio: float) -> np.ndarray:
    import pyclipper
    from shapely.geometry import Polygon

    poly = Polygon(box)
    distance = poly.area * ratio / max(poly.length, 1e-6)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    return np.array(expanded[0]) if expanded else box


class _Detector:
    def __init__(self, model_dir: Path, session_opts, limit: int):
        import onnxruntime as ort

        cfg = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))["PostProcess"]
        self.thresh, self.box_thresh = cfg["thresh"], cfg["box_thresh"]
        self.max_candidates, self.unclip_ratio = cfg["max_candidates"], cfg["unclip_ratio"]
        self.limit = limit
        self.sess = ort.InferenceSession(
            str(model_dir / "inference.onnx"), session_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, bgr: np.ndarray) -> List[np.ndarray]:
        h, w = bgr.shape[:2]
        ratio = self.limit / max(h, w) if max(h, w) > self.limit else 1.0
        rh = max(int(round(h * ratio / 32) * 32), 32)
        rw = max(int(round(w * ratio / 32) * 32), 32)
        img = cv2.resize(bgr, (rw, rh)).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        prob = self.sess.run(None, {self.input_name: x})[0][0, 0]

        bitmap = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours[: self.max_candidates]:
            box, side = _mini_box(contour)
            if side < 3:
                continue
            if _box_score(prob, box) < self.box_thresh:
                continue
            box, side = _mini_box(_unclip(box, self.unclip_ratio).reshape(-1, 1, 2).astype(np.float32))
            if side < 5:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / prob.shape[1] * w), 0, w)
            box[:, 1] = np.clip(np.round(box[:, 1] / prob.shape[0] * h), 0, h)
            boxes.append(box)
        return boxes


# ---------------------------------------------------------------- recognition
class _Recognizer:
    IMG_H, MIN_W = 48, 320

    def __init__(self, model_dir: Path, session_opts):
        import onnxruntime as ort

        cfg = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
        chars = [str(c) for c in cfg["PostProcess"]["character_dict"]]
        self.sess = ort.InferenceSession(
            str(model_dir / "inference.onnx"), session_opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        n_out = self.sess.get_outputs()[0].shape[-1]
        # index 0 = CTC blank, then the dictionary, then (optionally) a space.
        self.charset = ["<blank>"] + chars + ([" "] if isinstance(n_out, int) and n_out == len(chars) + 2 else [])

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        ratio = w / float(h)
        width = max(self.MIN_W, int(math.ceil(self.IMG_H * ratio)))
        new_w = min(width, int(math.ceil(self.IMG_H * ratio)))
        img = cv2.resize(crop, (new_w, self.IMG_H)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        canvas = np.zeros((self.IMG_H, width, 3), np.float32)
        canvas[:, :new_w] = img
        return canvas.transpose(2, 0, 1)[None]

    def __call__(self, crop: np.ndarray) -> Tuple[str, float]:
        probs = self.sess.run(None, {self.input_name: self._prepare(crop)})[0][0]
        idx = probs.argmax(axis=1)
        best = probs.max(axis=1)
        out, scores, prev = [], [], 0
        for i, p in zip(idx, best):
            if i != 0 and i != prev and i < len(self.charset):
                out.append(self.charset[i])
                scores.append(float(p))
            prev = i
        return "".join(out), (float(np.mean(scores)) if scores else 0.0)


def _perspective_crop(bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(box.astype(np.float32), std)
    crop = cv2.warpPerspective(bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / float(w) >= 1.5:                       # vertical text line
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


# --------------------------------------------------------------------- engine
class OnnxV5Engine(OCREngine):
    def __init__(self, config: Config):
        import onnxruntime as ort

        self.config = config
        dirs = ensure_models(Path(config.models_dir) if config.models_dir else _ROOT / "models")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.v5_cpu_threads
        opts.log_severity_level = 3                       # silence harmless graph-optimizer warnings
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.detector = _Detector(dirs["det"], opts, config.v5_det_limit)
        self.recognizer = _Recognizer(dirs["rec"], opts)

    def estimate_skew(self, image: Image.Image) -> float:
        """Median tilt (degrees, image coordinates) of the detected text lines; 0 if unclear.

        Positive = text slopes down to the right. ``image.rotate(angle)`` straightens it."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        angles = []
        for b in self.detector(bgr):                       # tl, tr, br, bl
            w = float(np.linalg.norm(b[1] - b[0]))
            h = float(np.linalg.norm(b[3] - b[0]))
            if w > 40 and w > 2.5 * h:
                angles.append(math.degrees(math.atan2(b[1][1] - b[0][1], b[1][0] - b[0][0])))
        if len(angles) < 3:
            return 0.0
        median = float(np.median(angles))
        return median if abs(median) <= 30 else 0.0

    def recognize_line(self, rgb: np.ndarray) -> Tuple[str, float]:
        """Recognize one already-cropped text line (RGB array)."""
        return self.recognizer(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def predict(self, image: Image.Image) -> List[OCRResult]:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        results: List[OCRResult] = []
        for box in self.detector(bgr):
            text, score = self.recognizer(_perspective_crop(bgr, box))
            text = visual_to_logical(text.strip())          # CTC output is in visual order
            if not text or score < self.config.min_confidence:
                continue
            results.append(OCRResult(
                page=0, text=text,
                bbox=[float(box[:, 0].min()), float(box[:, 1].min()),
                      float(box[:, 0].max()), float(box[:, 1].max())],
                confidence=score,
            ))
        return results

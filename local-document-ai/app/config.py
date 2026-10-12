"""Central configuration for the local Document AI pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Recognition engine: "onnx_v5"   = accurate PP-OCRv5 on onnxruntime (recommended),
    #                     "paddle_v5" = PP-OCRv5 via an isolated Paddle-3 worker process,
    #                     "paddle_v4" = fast PP-OCRv4, in-process.
    engine: str = "paddle_v4"        # also: "onnx_v5" (accurate, onnxruntime), "paddle_v5" (worker)
    models_dir: Optional[str] = None  # where ONNX models live; default <project>/models
    v5_python: Optional[str] = None  # interpreter with PaddlePaddle 3.x; default <project>/.venv_v5
    v5_cpu_threads: int = 8
    v5_det_limit: int = 1280         # max image side fed to the v5 text detector (speed/accuracy)

    # OCR (v4). Languages are all run; how their results are combined is merge_strategy.
    languages: List[str] = field(default_factory=lambda: ["ar", "en"])
    # "confidence" (best box per region) or "priority" (first language wins, others fill gaps)
    merge_strategy: str = "confidence"
    use_gpu: bool = False            # True -> NVIDIA GPU (needs paddlepaddle-gpu)
    min_confidence: float = 0.30     # drop detections below this score
    # PaddleOCR's Arabic model returns visual-order (reversed) text; convert it
    # to logical order. Set False if a future model already returns logical order.
    arabic_logical_order: bool = True

    # PDF rendering
    pdf_dpi: int = 300
    max_pages: int = 50

    # Preprocessing
    preprocess: bool = True
    upscale_target_long: int = 2400  # upscale small pages until the long side reaches this
    max_upscale: float = 4.0         # never enlarge more than this (1.0 disables upscaling)
    normalize_background: bool = True  # flatten paper/watermark patterns before OCR
    sharpen: bool = False

    # Layout / reading order
    line_overlap_ratio: float = 0.6  # box y-distance (x avg box height) to share a line
    dedupe_overlap: float = 0.3      # overlapping boxes from different language passes are merged

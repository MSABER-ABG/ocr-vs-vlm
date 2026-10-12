from app.ocr.base import OCREngine, OCRResult

__all__ = ["OCREngine", "OCRResult", "create_engine"]


def create_engine(config) -> OCREngine:
    """Factory: swap the recognition backend here without touching the pipeline."""
    if config.engine == "onnx_v5":
        from app.ocr.onnx_v5 import OnnxV5Engine

        return OnnxV5Engine(config)
    if config.engine == "paddle_v5":
        from app.ocr.paddle_v5 import PaddleV5Engine

        return PaddleV5Engine(config)
    if config.engine == "paddle_v4":
        from app.ocr.paddle_v4 import PaddleEngine

        return PaddleEngine(config)
    raise ValueError(f"unknown engine {config.engine!r} (use paddle_v4, onnx_v5 or paddle_v5)")

"""
OCR vs VLM Comparison App
Accord Business Group
Compare OCR (Tesseract / EasyOCR / PaddleOCR) vs GPT-4o Vision on PDF/image files
"""

import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import openai
import base64
import io
import re
import time
import json
import os
import shutil
import tempfile
import numpy as np
from pathlib import Path

# ─── Tesseract binary auto-detect (Windows) ──────────────────────────────────────
_TESSERACT_CANDIDATES = [
    shutil.which("tesseract"),
    r"C:/Program Files/Tesseract-OCR/tesseract.exe",
    r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    str(Path(__file__).parent / "OCR Software" / "tesseract.exe"),
]
for _cmd in _TESSERACT_CANDIDATES:
    if _cmd and Path(_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = _cmd
        break

# ─── EasyOCR lazy loader (cached so model loads only once) ───────────────────
@st.cache_resource(show_spinner=False)
def get_easyocr_reader(langs):
    """Load EasyOCR model once and cache it."""
    import easyocr
    return easyocr.Reader(list(langs), gpu=False)

# --- PaddleOCR lazy loader (cached so model loads only once) -----------------
@st.cache_resource(show_spinner=False)
def get_paddleocr_reader(lang):
    """Load a PaddleOCR model once per language and cache it.

    NOTE: On Windows, PaddlePaddle and PyTorch (used by EasyOCR) both ship
    their own copy of libiomp5md.dll, and Paddle registers its own DLL
    search directory at import time. If Paddle is imported before Torch in
    this process, Torch's shm.dll later fails to load with:
    "WinError 127: The specified procedure could not be found."
    Importing torch first (if it is installed) avoids the conflict, so we
    prime it here before ever importing paddleocr/paddle.
    """
    try:
        import torch  # noqa: F401  (import order guard, see note above)
    except ImportError:
        pass
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

# --- ABG OCR (local-document-ai) ------------------------------------------------
_HERE = Path(__file__).resolve().parent
LOCAL_DOC_AI_DIR = str(next((d for d in (_HERE / "local-document-ai", _HERE.parent / "local-document-ai")
                             if d.exists()), _HERE / "local-document-ai"))

@st.cache_resource(show_spinner=False)
def get_local_docai(langs, engine="paddle_v4"):
    """Build the local-document-ai pipeline once per (language set, engine)."""
    import sys
    if LOCAL_DOC_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DOC_AI_DIR)  # must precede this app.py so 'app' = the package
    from app.config import Config
    from app.pipeline import DocumentAI
    return DocumentAI(Config(languages=list(langs), engine=engine))

# ─── Arabic RTL character-order fix ───────────────────────────────────────────
# Tesseract has a well-known upstream limitation: on short/isolated RTL text
# blocks (e.g. single-word ID-card fields), it sometimes emits each Arabic
# word with its characters in reversed order, while digits/Latin text stay
# correct. This reverses each contiguous Arabic-script run back to proper
# reading order without touching numbers or Latin text.
_ARABIC_RUN = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")

def fix_arabic_rtl(text: str) -> str:
    return _ARABIC_RUN.sub(lambda m: m.group(0)[::-1], text)

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR vs VLM Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; color: #1a1a2e !important; }
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stSelectbox label, .stSlider label, .stTextInput label,
    .stTextArea label, .stFileUploader label { color: #1a1a2e !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a6e 0%, #3b3bbf 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select { color: #1a1a2e !important; background: white !important; }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: white !important; }

    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
        border-left: 4px solid #3b3bbf;
    }
    .result-card.vlm { border-left-color: #10b981; }

    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 6px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a6e; }
    .metric-label { font-size: 12px; color: #6b7280; margin-top: 4px; }

    .app-header {
        background: linear-gradient(135deg, #1a1a6e 0%, #3b3bbf 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 { color: #ffffff !important; margin: 0; font-size: 28px; text-shadow: 0 1px 3px rgba(0,0,0,0.3); }
    .app-header p { color: white !important; margin: 4px 0 0; }

    .badge-ocr {
        background: #dbeafe; color: #1e40af;
        padding: 4px 12px; border-radius: 999px;
        font-size: 13px; font-weight: 600; display: inline-block;
    }
    .badge-vlm {
        background: #d1fae5; color: #065f46;
        padding: 4px 12px; border-radius: 999px;
        font-size: 13px; font-weight: 600; display: inline-block;
    }
    .badge-tie {
        background: #fef3c7; color: #92400e;
        padding: 4px 12px; border-radius: 999px;
        font-size: 13px; font-weight: 600; display: inline-block;
    }

    [data-testid="metric-container"] { background: white; border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 OCR vs VLM")
    st.markdown("**Comparison Lab**")
    st.markdown("---")

    st.markdown("### ⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your OpenAI API key — never stored",
        placeholder="sk-..."
    )

    st.markdown("---")
    st.markdown("### 🔤 OCR Engine")

    ocr_engine = st.selectbox(
        "OCR Engine",
        options=[
            "Tesseract OCR",
            "Easy OCR",
            "Paddle OCR",
            "ABG OCR",
        ],
        index=0,
    )

    # Language selection depends on engine
    ocr_lang = None
    easyocr_langs = None
    paddle_langs = None
    local_langs = None
    id_mode = False
    cheque_mode = False
    local_engine = "paddle_v4"
    fix_rtl = False

    if ocr_engine == "Tesseract OCR":
        ocr_lang = st.selectbox(
            "Language",
            options=["ara", "eng", "ara+eng", "osd"],
            index=0,
            help="Tesseract language pack"
        )
        fix_rtl = st.checkbox(
            "Fix reversed Arabic characters",
            value=True,
            help="Tesseract sometimes reverses the character order of short/isolated "
                 "Arabic words (e.g. ID-card fields). Enable to auto-correct."
        )
    elif ocr_engine == "Easy OCR":
        ocr_lang_display = st.selectbox(
            "Language",
            options=["Arabic", "English", "Arabic + English"],
            index=0,
            help="Language selection"
        )
        LANG_MAP = {
            "Arabic": ["ar"],
            "English": ["en"],
            "Arabic + English": ["ar", "en"],
        }
        easyocr_langs = LANG_MAP[ocr_lang_display]
    elif ocr_engine == "ABG OCR":
        ocr_lang_display = st.selectbox(
            "Language",
            options=["Arabic + English", "Arabic", "English"],
            index=0,
            help="First language is primary; the others only fill regions it missed. "
                 "Use English only for English documents."
        )
        local_langs = {
            "Arabic": ["ar"],
            "English": ["en"],
            "Arabic + English": ["ar", "en"],
        }[ocr_lang_display]
        try:
            import onnxruntime  # noqa: F401
            _v5_ready = True
        except Exception:
            _v5_ready = False
        _speed = st.selectbox(
            "Recognition model",
            options=["Accurate - PP-OCRv5", "Fast - PP-OCRv4"],
            index=0 if _v5_ready else 1,
            help="PP-OCRv5 reads Arabic noticeably better (names, cheque text, digits). "
                 "It runs locally on CPU via ONNX; the models (~100 MB) are downloaded "
                 "once on first use.",
        )
        local_engine = "onnx_v5" if _speed.startswith("Accurate") else "paddle_v4"
        if local_engine == "onnx_v5" and not _v5_ready:
            st.warning("onnxruntime is not installed - falling back to Fast.")
            local_engine = "paddle_v4"
        _doc = st.selectbox(
            "Document type",
            options=["General", "Egyptian ID card", "Cheque"],
            index=0,
            help="ID card: crops/straightens the card, reads the 14-digit national number digit by "
                 "digit and validates it. Cheque: finds the date, payee and amounts next to their "
                 "labels and cross-checks the amount in words against the digits. "
                 "Parsed fields appear under 'Structured output'."
        )
        id_mode = _doc == "Egyptian ID card"
        cheque_mode = _doc == "Cheque"
    else:  # PaddleOCR
        ocr_lang_display = st.selectbox(
            "Language",
            options=["Arabic", "English", "Arabic + English"],
            index=0,
            help="Language selection"
        )
        PADDLE_LANG_MAP = {
            "Arabic": ["ar"],
            "English": ["en"],
            "Arabic + English": ["ar", "en"],
        }
        paddle_langs = PADDLE_LANG_MAP[ocr_lang_display]
        fix_rtl = st.checkbox(
            "Fix reversed Arabic characters",
            value=False,
            help="Enable only if the output looks character-reversed for you."
        )

    enhance_image = st.checkbox(
        "🔍 Enhance image before OCR",
        value=True,
        help="Upscales, sharpens and boosts contrast before running the OCR engine. "
             "Recommended for small/low-quality scans (e.g. ID cards) — does not "
             "affect the VLM, which already sees the original image."
    )

    st.markdown("---")

    vlm_model = st.selectbox(
        "VLM Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0
    )

    st.markdown("---")
    st.markdown("**ABG · Accord Business Group**")
    st.markdown("*Local → Cloud Pipeline*")

# Fixed defaults
dpi = 300
max_pages = 3
vlm_prompt = (
    "Extract all text and data from this image. "
    "Return as valid JSON with keys: "
    "\"text\" (full text), \"fields\" (key-value pairs found), "
    "\"tables\" (any tabular data), \"language\" (detected language), "
    "\"confidence\" (your confidence 0-100), "
    "\"notes\" (any observations)."
)

# ─── Header ──────────────────────────────────────────────────────────────────────
engine_label = ocr_engine
st.markdown(f"""
<div class="app-header">
    <h1>🔬 OCR vs VLM — Live Comparison</h1>
    <p>Upload a PDF or image · Run {engine_label} &amp; GPT-4o Vision · Compare side-by-side</p>
</div>
""", unsafe_allow_html=True)

# ─── File Upload ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📎 Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
    help="PDF will be converted page-by-page. Images processed directly."
)

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def render_pdf_pages(file_bytes: bytes, dpi: int, max_pages: int) -> list:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages = pdf2image.convert_from_path(tmp_path, dpi=dpi)
        return pages[:max_pages]
    finally:
        os.unlink(tmp_path)


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def preprocess_for_ocr(img: Image.Image, upscale: float = 2.0) -> Image.Image:
    """Enhance a scanned/photographed document for better OCR accuracy.

    Small, dense text (e.g. ID-card fields) is the hardest case for general
    purpose OCR engines. Upscaling, converting to grayscale, boosting
    contrast and sharpening measurably improves character recognition
    without needing any extra dependency (pure Pillow). This is only
    applied to the OCR path — the VLM already receives the original image.
    """
    from PIL import ImageOps, ImageFilter

    if upscale and upscale != 1.0:
        w, h = img.size
        img = img.resize((int(w * upscale), int(h * upscale)), Image.LANCZOS)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    sharpened = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return sharpened.convert("RGB")


def run_tesseract(images: list, lang: str, fix_rtl: bool = False) -> dict:
    """Run Tesseract OCR — fast, works offline."""
    start = time.time()
    texts = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img, lang=lang)
            if fix_rtl and "ara" in lang:
                text = fix_arabic_rtl(text)
            texts.append(text)
        except Exception as e:
            texts.append(f"[Error on page: {e}]")
    elapsed = time.time() - start
    full_text = "\n\n--- Page Break ---\n\n".join(texts)
    return {
        "text": full_text,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"Tesseract OCR ({lang})",
    }


def run_easyocr(images: list, lang_codes: list, fix_rtl: bool = False) -> dict:
    """Run EasyOCR — better Arabic accuracy, offline."""
    start = time.time()
    reader = get_easyocr_reader(tuple(lang_codes))
    texts = []
    for img in images:
        try:
            img_np = np.array(img)
            results = reader.readtext(img_np, detail=0, paragraph=True)
            line = "\n".join(results)
            if fix_rtl:
                line = fix_arabic_rtl(line)
            texts.append(line)
        except Exception as e:
            texts.append(f"[Error on page: {e}]")
    elapsed = time.time() - start
    full_text = "\n\n--- Page Break ---\n\n".join(texts)
    return {
        "text": full_text,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"Easy OCR ({', '.join(lang_codes)})",
    }


def group_text_boxes_into_lines(detections: list, y_overlap_ratio: float = 0.6) -> list:
    """Group PaddleOCR's per-word/box detections into visual text lines.

    PaddleOCR detects and recognizes each text box independently (unlike
    Tesseract/EasyOCR, which do their own layout analysis), so a form with
    one word per box comes back as one "line" per word. This groups boxes
    that sit on roughly the same horizontal band into a single line.

    Within each line, boxes are ordered by reading direction:
      - Lines containing Arabic script are ordered right-to-left (by each
        box's right edge), matching Arabic reading order.
      - Lines that are purely digits/punctuation (phone numbers, ID
        numbers, serials) are ordered left-to-right (by each box's left
        edge) — numbers keep their natural left-to-right order even inside
        an otherwise right-to-left document.

    ``detections`` is PaddleOCR's raw ``result[0]``: a list of
    ``[box, (text, confidence)]`` where ``box`` is 4 (x, y) corner points.
    """
    if not detections:
        return []

    items = []
    heights = []
    for box, (text, _conf) in detections:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        height = max(ys) - min(ys)
        items.append({
            "text": text,
            "y": sum(ys) / len(ys),
            "x_min": min(xs),
            "x_max": max(xs),
        })
        heights.append(height)

    avg_h = sum(heights) / len(heights) if heights else 20
    threshold = max(avg_h * y_overlap_ratio, 5)

    items.sort(key=lambda it: it["y"])

    lines = [[items[0]]]
    for it in items[1:]:
        # Compare against the running average y of the current line so
        # slight skew doesn't cause drift across many boxes.
        line_y = sum(x["y"] for x in lines[-1]) / len(lines[-1])
        if abs(it["y"] - line_y) <= threshold:
            lines[-1].append(it)
        else:
            lines.append([it])

    line_strings = []
    for line in lines:
        has_arabic = any(_ARABIC_RUN.search(it["text"]) for it in line)
        if has_arabic:
            line.sort(key=lambda it: it["x_max"], reverse=True)  # right-to-left
        else:
            line.sort(key=lambda it: it["x_min"])  # left-to-right (numbers)
        line_strings.append(" ".join(it["text"] for it in line))
    return line_strings


def run_paddleocr(images: list, lang_codes: list, fix_rtl: bool = False) -> dict:
    """Run PaddleOCR — high accuracy, offline, strong Arabic script support."""
    start = time.time()
    readers = [get_paddleocr_reader(code) for code in lang_codes]
    texts = []
    for img in images:
        img_np = np.array(img.convert("RGB"))
        all_detections = []
        errors = []
        for reader in readers:
            try:
                result = reader.ocr(img_np, cls=True)
                if result and result[0]:
                    all_detections.extend(result[0])
            except Exception as e:
                errors.append(f"[Error on page: {e}]")
        page_lines = group_text_boxes_into_lines(all_detections) + errors
        page_text = "\n".join(page_lines)
        if fix_rtl:
            page_text = fix_arabic_rtl(page_text)
        texts.append(page_text)
    elapsed = time.time() - start
    full_text = "\n\n--- Page Break ---\n\n".join(texts)
    return {
        "text": full_text,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"Paddle OCR ({', '.join(lang_codes)})",
    }


def run_local_id_card(images: list, lang_codes: list, engine: str = "paddle_v4") -> dict:
    """Egyptian ID card template: one result per image (each image = one card)."""
    start = time.time()
    docai = get_local_docai(tuple(lang_codes), engine)
    import sys
    if LOCAL_DOC_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DOC_AI_DIR)
    from app.templates.egypt_id import EgyptIDReader
    reader = EgyptIDReader(docai)

    cards, blocks = [], []
    for img in images:
        res = reader.read(img)
        cards.append(res.fields)
        # Plain text only (no labels): the card's text top to bottom.
        blocks.append(chr(10).join(res.text_lines))
    full_text = (chr(10) * 2 + "--- Page Break ---" + chr(10) * 2).join(blocks)
    elapsed = time.time() - start
    return {
        "text": full_text,
        "structured": cards if len(cards) > 1 else cards[0],
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"ABG OCR · Egyptian ID · {engine[-2:].upper()} ({', '.join(lang_codes)})",
    }


def run_local_cheque(images: list, lang_codes: list, engine: str = "paddle_v4") -> dict:
    """Cheque template: label-anchored fields + cross-checks. One result per image."""
    start = time.time()
    docai = get_local_docai(tuple(lang_codes), engine)
    import sys
    if LOCAL_DOC_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DOC_AI_DIR)
    from app.templates.cheque import ChequeReader
    reader = ChequeReader(docai)

    cheques, blocks = [], []
    for img in images:
        res = reader.read(img)
        cheques.append(res.fields)
        blocks.append(chr(10).join(res.text_lines))
    full_text = (chr(10) * 2 + "--- Page Break ---" + chr(10) * 2).join(blocks)
    return {
        "text": full_text,
        "structured": cheques if len(cheques) > 1 else cheques[0],
        "expand_structured": True,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(time.time() - start, 2),
        "cost_usd": 0.0,
        "engine": f"ABG OCR · Cheque · {engine[-2:].upper()} ({', '.join(lang_codes)})",
    }


def run_local_docai(images: list, lang_codes: list, engine: str = "paddle_v4") -> dict:
    """Run the local-document-ai pipeline (own preprocessing, lines + RTL/LTR order)."""
    start = time.time()
    docai = get_local_docai(tuple(lang_codes), engine)
    doc = docai.process_pages(images, source="upload")
    from app.output import to_text, to_dict
    full_text = to_text(doc)
    elapsed = time.time() - start
    return {
        "text": full_text,
        "structured": to_dict(doc),
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"ABG OCR · {engine[-2:].upper()} ({', '.join(lang_codes)})",
    }


def run_ocr_engine(images: list, engine: str, lang: str, easyocr_langs: list,
                    paddle_langs: list = None, fix_rtl: bool = False,
                    local_langs: list = None, id_mode: bool = False,
                    local_engine: str = "paddle_v4", cheque_mode: bool = False) -> dict:
    """Dispatch to the selected OCR engine."""
    if engine == "ABG OCR":
        if cheque_mode:
            return run_local_cheque(images, local_langs, local_engine)
        if id_mode:
            return run_local_id_card(images, local_langs, local_engine)
        return run_local_docai(images, local_langs, local_engine)
    if engine == "Easy OCR":
        return run_easyocr(images, easyocr_langs, fix_rtl)
    elif engine == "Paddle OCR":
        return run_paddleocr(images, paddle_langs, fix_rtl)
    else:
        return run_tesseract(images, lang, fix_rtl)


def run_vlm(images: list, api_key: str, model: str, prompt: str) -> dict:
    import httpx
    client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
    start = time.time()
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, img in enumerate(images):
        b64 = image_to_base64(img)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
            ]
        }]
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                response_format={"type": "json_object"}, max_tokens=4000
            )
            content = resp.choices[0].message.content
            total_input_tokens += resp.usage.prompt_tokens
            total_output_tokens += resp.usage.completion_tokens
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_response": content, "parse_error": True}
            parsed["_page"] = i + 1
            all_results.append(parsed)
        except Exception as e:
            all_results.append({"_page": i + 1, "error": str(e)})

    elapsed = time.time() - start
    if model == "gpt-4o":
        cost = (total_input_tokens / 1_000_000 * 2.50) + (total_output_tokens / 1_000_000 * 10.00)
    else:
        cost = (total_input_tokens / 1_000_000 * 0.15) + (total_output_tokens / 1_000_000 * 0.60)

    combined_text = ""
    for r in all_results:
        combined_text += r.get("text", r.get("raw_response", "")) + "\n\n"

    return {
        "results": all_results,
        "combined_text": combined_text.strip(),
        "pages": len(images),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": round(cost, 4),
        "time_sec": round(elapsed, 2),
        "model": model,
    }


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def build_comparison_table(ocr_result: dict, vlm_result: dict) -> dict:
    eng = ocr_result.get("engine", "OCR")
    arabic_note = "✅ Native Arabic" if ("Easy OCR" in eng or "Paddle OCR" in eng or "ABG OCR" in eng) else "⚠️ Requires ara pack"
    return {
        "Speed (sec)": {
            "OCR": ocr_result["time_sec"],
            "VLM": vlm_result["time_sec"],
            "Winner": "OCR" if ocr_result["time_sec"] < vlm_result["time_sec"] else "VLM"
        },
        "Cost (USD)": {
            "OCR": f"${ocr_result['cost_usd']:.4f}",
            "VLM": f"${vlm_result['cost_usd']:.4f}",
            "Winner": "OCR"
        },
        "Words Extracted": {
            "OCR": count_words(ocr_result["text"]),
            "VLM": count_words(vlm_result["combined_text"]),
            "Winner": "VLM" if count_words(vlm_result["combined_text"]) > count_words(ocr_result["text"]) else "OCR"
        },
        "Structured Output": {
            "OCR": "❌ Raw text only",
            "VLM": "✅ JSON with fields/tables",
            "Winner": "VLM"
        },
        "Arabic Support": {
            "OCR": arabic_note,
            "VLM": "✅ Native multilingual",
            "Winner": "VLM"
        },
        "Offline Use": {
            "OCR": "✅ Works offline",
            "VLM": "❌ Needs internet",
            "Winner": "OCR"
        },
        "Tokens Used": {
            "OCR": "N/A",
            "VLM": f"{vlm_result['input_tokens']:,} in / {vlm_result['output_tokens']:,} out",
            "Winner": "OCR"
        },
    }


def show_decision_framework():
    st.markdown("### 🧭 Decision Framework")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="result-card">
<h4>✅ Use OCR (offline) when:</h4>
<ul>
<li>High volume, low cost is priority</li>
<li>Offline / air-gapped environment</li>
<li>Printed text (no complex tables)</li>
<li>No API key / internet required</li>
<li><b>Easy OCR</b>: better Arabic accuracy</li>
<li><b>Tesseract OCR</b>: fastest, lightest</li>
</ul>
<span class="badge-ocr">Best for: Bulk text extraction</span>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="result-card vlm">
<h4>✅ Use VLM (GPT-4o Vision) when:</h4>
<ul>
<li>Forms with key-value pairs to extract</li>
<li>Tables, charts, or structured data</li>
<li>Low volume, high accuracy needed</li>
<li>Semantic understanding required</li>
<li>Output must be structured JSON</li>
<li>Handwritten or noisy documents</li>
</ul>
<span class="badge-vlm">Best for: Banking forms, contracts, reports</span>
</div>
""", unsafe_allow_html=True)


# ─── Main Logic ──────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("---")
    show_decision_framework()
else:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()

    with st.spinner("📄 Preparing file..."):
        if file_ext == ".pdf":
            try:
                images = render_pdf_pages(file_bytes, dpi, max_pages)
                st.success(f"✅ PDF rendered: {len(images)} page(s) at {dpi} DPI")
            except Exception as e:
                st.error(f"PDF render error: {e}")
                st.stop()
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                images = [img]
                st.success(f"✅ Image loaded: {img.size[0]}×{img.size[1]}px")
            except Exception as e:
                st.error(f"Image load error: {e}")
                st.stop()

    with st.expander("🖼️ Preview pages", expanded=False):
        cols = st.columns(min(len(images), 3))
        for i, (img, col) in enumerate(zip(images, cols)):
            col.image(img, caption=f"Page {i+1}")

    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    run_ocr = col_btn1.button("⚡ Run OCR Only", use_container_width=True, type="secondary")
    run_vlm_only = col_btn2.button("🤖 Run VLM Only", use_container_width=True, type="secondary",
                                    disabled=not api_key)
    run_both = col_btn3.button("🔬 Run BOTH & Compare", use_container_width=True, type="primary",
                                disabled=not api_key)

    if not api_key and (run_vlm_only or run_both):
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to use VLM.")

    if "ocr_result" not in st.session_state:
        st.session_state.ocr_result = None
    if "vlm_result" not in st.session_state:
        st.session_state.vlm_result = None

    if run_ocr or run_both:
        spinner_msg = f"⚡ Running {ocr_engine}... first run loads the models, please wait"
        with st.spinner(spinner_msg):
            try:
                own_preprocessing = ocr_engine == "ABG OCR"
                ocr_images = (
                    [preprocess_for_ocr(img) for img in images]
                    if enhance_image and not own_preprocessing else images
                )
                st.session_state.ocr_result = run_ocr_engine(
                    ocr_images, ocr_engine, ocr_lang, easyocr_langs, paddle_langs, fix_rtl,
                    local_langs, id_mode, local_engine, cheque_mode
                )
                r = st.session_state.ocr_result
                st.success(f"✅ OCR done in {r['time_sec']}s — {r['engine']}")
            except Exception as e:
                st.error(f"OCR error: {e}")
                st.session_state.ocr_result = None

    if run_vlm_only or run_both:
        if api_key:
            with st.spinner(f"🤖 Running {vlm_model} Vision... (~5-20s per page)"):
                try:
                    st.session_state.vlm_result = run_vlm(images, api_key, vlm_model, vlm_prompt)
                    r = st.session_state.vlm_result
                    st.success(f"✅ VLM done in {r['time_sec']}s · Cost: ${r['cost_usd']:.4f}")
                except Exception as e:
                    st.error(f"VLM error: {e}")
                    st.session_state.vlm_result = None

    ocr_res = st.session_state.ocr_result
    vlm_res = st.session_state.vlm_result

    if ocr_res or vlm_res:
        st.markdown("---")

        if ocr_res and vlm_res:
            st.markdown("### 📊 Head-to-Head Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ OCR Speed", f"{ocr_res['time_sec']}s")
            m2.metric("🤖 VLM Speed", f"{vlm_res['time_sec']}s")
            m3.metric("💰 OCR Cost", "$0.0000")
            m4.metric("💰 VLM Cost", f"${vlm_res['cost_usd']:.4f}")
            st.markdown("---")

        left, right = st.columns(2)

        with left:
            engine_label_display = ocr_engine
            engine_subtitle = ""
            st.markdown(f"""
<div style="background:#dbeafe;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
<b style="color:#1e40af;font-size:16px;">⚡ {engine_label_display}</b><br>
<span style="color:#3b82f6;font-size:12px;">{engine_subtitle}</span>
</div>
""", unsafe_allow_html=True)

            if ocr_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Time", f"{ocr_res['time_sec']}s")
                c2.metric("Words", f"{ocr_res['word_count']:,}")
                c3.metric("Cost", "$0.00")
                st.text_area("Raw OCR Text", value=ocr_res["text"], height=400, key="ocr_text_display")
                st.download_button("⬇️ Download OCR Text", data=ocr_res["text"],
                                   file_name="ocr_output.txt", mime="text/plain")
                if ocr_res.get("structured"):
                    structured_json = json.dumps(ocr_res["structured"], ensure_ascii=False, indent=2)
                    with st.expander("🧱 Structured output", expanded=bool(ocr_res.get("expand_structured"))):
                        st.code(structured_json, language="json")
                    st.download_button("⬇️ Download Structured JSON", data=structured_json,
                                       file_name="document_ai.json", mime="application/json")
            else:
                st.info("Run OCR to see results here.")

        with right:
            st.markdown("""
<div style="background:#d1fae5;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
<b style="color:#065f46;font-size:16px;">🤖 VLM — GPT-4o Vision</b><br>
<span style="color:#10b981;font-size:12px;">Structured · Multilingual · JSON</span>
</div>
""", unsafe_allow_html=True)

            if vlm_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Time", f"{vlm_res['time_sec']}s")
                c2.metric("Tokens", f"{vlm_res['input_tokens'] + vlm_res['output_tokens']:,}")
                c3.metric("Cost", f"${vlm_res['cost_usd']:.4f}")
                tab_json, tab_text, tab_raw = st.tabs(["📋 JSON", "📝 Text", "🔢 Raw"])
                with tab_json:
                    for page_result in vlm_res["results"]:
                        page_num = page_result.get("_page", "?")
                        with st.expander(f"Page {page_num}", expanded=True):
                            st.json({k: v for k, v in page_result.items() if k != "_page"})
                with tab_text:
                    st.text_area("Extracted Text (from VLM)", value=vlm_res["combined_text"],
                                 height=400, key="vlm_text_display")
                with tab_raw:
                    st.code(json.dumps(vlm_res["results"], ensure_ascii=False, indent=2), language="json")
                json_str = json.dumps(vlm_res["results"], ensure_ascii=False, indent=2)
                st.download_button("⬇️ Download VLM JSON", data=json_str,
                                   file_name="vlm_output.json", mime="application/json")
            else:
                st.info("Run VLM to see results here.")

        if ocr_res and vlm_res:
            st.markdown("---")
            st.markdown("### 📋 Detailed Comparison")
            comp = build_comparison_table(ocr_res, vlm_res)
            rows = []
            for metric, vals in comp.items():
                rows.append(
                    f"<tr>"
                    f"<td><b>{metric}</b></td>"
                    f"<td>{vals['OCR']}</td>"
                    f"<td>{vals['VLM']}</td>"
                    f"</tr>"
                )
            table_html = f"""
<table style="width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
<thead>
<tr style="background:#1a1a6e;color:white;">
<th style="padding:12px 16px;text-align:left;">Metric</th>
<th style="padding:12px 16px;text-align:left;">⚡ OCR</th>
<th style="padding:12px 16px;text-align:left;">🤖 VLM</th>
</tr>
</thead>
<tbody>
{"".join(f'<tr style="border-bottom:1px solid #f0f0f0;">' + row[4:] for row in rows)}
</tbody>
</table>
"""
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown("---")
            ocr_wins = sum(1 for v in comp.values() if v["Winner"] == "OCR")
            vlm_wins = sum(1 for v in comp.values() if v["Winner"] == "VLM")
            if vlm_wins > ocr_wins:
                st.success(f"🏆 **VLM wins {vlm_wins}-{ocr_wins}** for this document.")
            elif ocr_wins > vlm_wins:
                st.info(f"🏆 **OCR wins {ocr_wins}-{vlm_wins}** for this document.")
            else:
                st.warning(f"🤝 **Tie {ocr_wins}-{vlm_wins}** — choose based on priority.")

        st.markdown("---")
        show_decision_framework()

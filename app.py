"""
OCR Platform
Accord Business Group
Compare OCR (Tesseract / EasyOCR / PaddleOCR) vs GPT-4o Vision on PDF/image files
"""

import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import pdf2image
import openai
import base64
import io
import re
import time
import hashlib
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

# --- Online model ids ------------------------------------------------------------
# The API wants the exact lowercase id ("gpt-4o"), never a display name ("GPT-4o").
# The ONLY online models offered in the list (shown in this order).
_DEFAULT_ONLINE_MODELS = ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna", "gpt-4o", "gpt-4o-mini"]
# The ONLY model treated as the correct answer in the comparison tab (also the default online choice).
REFERENCE_MODEL = "gpt-6.1-sol"
_NON_CHAT = ("embedding", "audio", "tts", "whisper", "realtime", "transcribe", "image", "moderation",
             "search", "instruct", "dall-e", "davinci", "babbage", "codex", "computer-use", "diarize")


def normalize_model_id(model: str) -> str:
    """'GPT-4o' / 'gpt 4o' -> 'gpt-4o'. Fine-tunes (ft:...) and other ids are left untouched."""
    m = (model or "").strip()
    if m.lower().startswith(("gpt", "o1", "o3", "o4", "chatgpt")) and not m.lower().startswith("ft:"):
        m = "-".join(m.lower().split())
    return m


@st.cache_data(show_spinner=False, ttl=600)
def _list_account_models(key_hash: str, api_key: str) -> list:
    import httpx
    client = openai.OpenAI(api_key=api_key, http_client=httpx.Client(), timeout=15)
    ids = [m.id for m in client.models.list()]
    keep = [i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))
            and not any(k in i for k in _NON_CHAT) and not re.search(r"-(\d{4}-\d{2}-\d{2}|\d{4})$", i)]
    return sorted(set(keep))


def _online_model_ids(api_key: str) -> list:
    """The five offered models; with a key, only those this account can really use (all five if the
    account list cannot be read or has none of them)."""
    if api_key:
        try:
            ids = set(_list_account_models(hashlib.sha256(api_key.encode()).hexdigest()[:12], api_key))
            usable = [m for m in _DEFAULT_ONLINE_MODELS if m in ids]
            if usable:
                return usable
        except Exception:
            pass
    return list(_DEFAULT_ONLINE_MODELS)


def _summary_mod():
    """Key-field table logic (lives in the engine package)."""
    import sys
    if LOCAL_DOC_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DOC_AI_DIR)
    from app.templates import summary
    return summary


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
    page_title="OCR Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Cairo:wght@400;600;700&display=swap');

    :root {
        --ink: #0f172a;  --muted: #64748b;  --line: #e2e8f0;
        --brand: #4f46e5; --brand-2: #7c3aed; --accent: #06b6d4; --ok: #10b981;
        --surface: #ffffff; --bg: #f4f6fb;
    }

    /* ---- base ---------------------------------------------------------- */
    html, body, .stApp, [class*="css"] { font-family: 'Inter', 'Cairo', 'Segoe UI', system-ui, sans-serif; }
    .stApp {
        background:
            radial-gradient(1100px 500px at 100% -10%, rgba(124,58,237,0.10), transparent 60%),
            radial-gradient(900px 500px at -10% 0%, rgba(6,182,212,0.08), transparent 55%),
            var(--bg);
        color: var(--ink) !important;
    }
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stSelectbox label, .stSlider label, .stTextInput label,
    .stTextArea label, .stFileUploader label { color: var(--ink) !important; }
    .block-container { max-width: 1320px; padding-top: 1.4rem; padding-bottom: 3rem; }
    h3 { font-weight: 700; letter-spacing: -0.01em; }

    /* hide Streamlit chrome that does not belong to a product UI */
    [data-testid="stDecoration"], [data-testid="stDeployButton"], footer { display: none !important; }
    [data-testid="stHeader"] { background: transparent; }

    /* ---- sidebar ------------------------------------------------------------ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1020 0%, #151a3d 55%, #24206b 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    /* .stApp prefix: beats the generic ".stApp p/span/label" dark-text rule above */
    .stApp [data-testid="stSidebar"], .stApp [data-testid="stSidebar"] *,
    .stApp [data-testid="stSidebar"] p, .stApp [data-testid="stSidebar"] span,
    .stApp [data-testid="stSidebar"] label, .stApp [data-testid="stSidebar"] div { color: #ffffff !important; }
    [data-testid="stSidebarUserContent"] { padding: 0.6rem 1rem 0.5rem 1rem !important; }
    [data-testid="stSidebarHeader"] { height: 2rem !important; padding: 0.4rem 1rem !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: clamp(0.2rem, 1.7vh, 1.6rem) !important; justify-content: space-between; }
    /* spread over the whole sidebar height; footer pinned to the bottom */
    [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"],
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]:first-child { min-height: min(calc(100vh - 4.6rem), 860px); }
    /* blocks inside columns (the API key row) must NOT inherit the full-height spread above */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { align-items: flex-end !important; gap: 0.4rem !important; flex: 0 0 auto !important; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] { min-height: 0 !important; justify-content: flex-start !important; gap: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button { min-height: 30px !important; height: 30px !important; width: 34px !important; padding: 0 !important; font-size: 13px; margin-bottom: 4px; border-radius: 9px; }
    /* the send button takes the sidebar colour (thin border like the input boxes, so it stays visible) */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button { background: #151a3d !important; border: 1px solid rgba(255,255,255,0.16) !important; box-shadow: none !important; color: #ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button:hover { border-color: #a5b4fc !important; transform: none; }
    [data-testid="stSidebar"] [data-testid="InputInstructions"] { display: none !important; }      /* hide "Press Enter to apply" */
    [data-testid="stSidebar"] .element-container:has(h3), [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(h3) { margin-bottom: -0.55rem !important; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { margin-bottom: 0 !important; min-height: 0 !important; }
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stCheckbox { margin-bottom: 0 !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div { min-height: 38px !important; height: 38px !important; align-items: center !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div > div { padding-top: 0 !important; padding-bottom: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] { width: 92% !important; }
    [data-testid="stSidebar"] input { padding-top: 6px !important; padding-bottom: 6px !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
    [data-testid="stSidebar"] h3, .stApp [data-testid="stSidebar"] .stMarkdown h3 {
        margin: 0 !important; padding: 8px 0 12px 0 !important; line-height: 1.3 !important; height: auto !important; }
    [data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] { margin: 0 !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 11px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
        margin: 8px 0 0 0 !important; padding: 0 !important; opacity: .75;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10) !important; margin: 6px 0 !important; }
    [data-testid="stSidebar"] label p { font-size: 13px; font-weight: 500; }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        border-radius: 12px !important;
        transition: border-color .15s ease, background .15s ease;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
    [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
        border-color: #818cf8 !important; background: rgba(255,255,255,0.11) !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] input { color: #ffffff !important; background: transparent !important; }
    [data-testid="stSidebar"] div[data-baseweb="base-input"] { background: transparent !important; }
    [data-testid="stSidebar"] svg { fill: #c7d2fe !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"], .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { font-size: 11.5px; line-height: 1.35; }

    .brand { display: flex; align-items: center; gap: 12px; padding: 0 2px 14px 2px; }
    .brand .mark {
        width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
        background: linear-gradient(135deg, #6366f1, #a855f7 60%, #06b6d4);
        box-shadow: 0 6px 18px rgba(99,102,241,0.45); font-weight: 800; font-size: 13px; color: #fff !important;
        letter-spacing: 0.02em;
    }
    .brand .name { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; color: #fff !important; line-height: 1.1; }
    .brand .tag  { font-size: 11px; opacity: .75; margin-top: 2px; }
    .side-foot { margin-top: 4px; padding: 8px 12px; border-radius: 12px; background: rgba(255,255,255,0.05);
                 border: 1px solid rgba(255,255,255,0.08); }
    .side-foot b { font-size: 12px; color: #fff !important; }
    .side-foot span { display: block; font-size: 11px; opacity: .75; margin-top: 1px; }

    /* ---- hero ---------------------------------------------------------------- */
    .app-header {
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, #0f172a 0%, #312e81 48%, #6d28d9 100%);
        padding: 30px 36px; border-radius: 22px; margin-bottom: 22px;
        box-shadow: 0 18px 40px rgba(49,46,129,0.28);
    }
    .app-header::before {
        content: ""; position: absolute; right: -80px; top: -120px; width: 380px; height: 380px; border-radius: 50%;
        background: radial-gradient(circle, rgba(6,182,212,0.55), transparent 65%);
    }
    .app-header::after {
        content: ""; position: absolute; left: 30%; bottom: -160px; width: 360px; height: 360px; border-radius: 50%;
        background: radial-gradient(circle, rgba(168,85,247,0.40), transparent 65%);
    }
    .stApp .app-header, .stApp .app-header * { color: #ffffff !important; position: relative; z-index: 1; }
    .app-header h1 { margin: 0; font-size: 34px; font-weight: 800; letter-spacing: -0.02em; }
    .app-header p  { margin: 8px 0 14px 0; font-size: 14.5px; opacity: .85; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
        padding: 5px 12px; border-radius: 999px; font-size: 12px; font-weight: 600;
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25); backdrop-filter: blur(6px);
    }

    /* ---- cards / badges ----------------------------------------------------------- */
    .result-card {
        background: var(--surface); border-radius: 18px; padding: 22px 24px; margin-bottom: 16px;
        border: 1px solid var(--line); box-shadow: 0 6px 24px rgba(15,23,42,0.06);
        border-top: 4px solid var(--brand);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .result-card:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(15,23,42,0.10); }
    .result-card.vlm { border-top-color: var(--ok); }
    .result-card h4 { margin-top: 0; font-weight: 700; }
    .metric-box { background: var(--surface); border-radius: 14px; padding: 16px; text-align: center;
                  box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin: 6px; }
    .metric-value { font-size: 28px; font-weight: 700; color: var(--brand); }
    .metric-label { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .badge-ocr, .badge-vlm, .badge-tie { padding: 5px 13px; border-radius: 999px; font-size: 12.5px;
                                         font-weight: 600; display: inline-block; }
    .badge-ocr { background: #e0e7ff; color: #3730a3; }
    .badge-vlm { background: #d1fae5; color: #065f46; }
    .badge-tie { background: #fef3c7; color: #92400e; }

    .panel-title {
        display: flex; align-items: center; gap: 10px; padding: 12px 16px; margin-bottom: 14px;
        border-radius: 14px; font-weight: 700; font-size: 15px; background: var(--surface);
        border: 1px solid var(--line); box-shadow: 0 4px 14px rgba(15,23,42,0.05);
    }
    .panel-title .dot { width: 10px; height: 10px; border-radius: 50%; }
    .panel-title.ocr .dot { background: var(--brand); box-shadow: 0 0 0 4px rgba(79,70,229,0.18); }
    .panel-title.vlm .dot { background: var(--ok);    box-shadow: 0 0 0 4px rgba(16,185,129,0.20); }
    .panel-title small { margin-left: auto; font-weight: 600; color: var(--muted) !important; font-size: 11px; }

    /* ---- widgets ------------------------------------------------------------------ */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface); border: 2px dashed #c7d2fe; border-radius: 18px; padding: 22px;
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--brand-2); box-shadow: 0 8px 24px rgba(124,58,237,0.12); }
    [data-testid="stFileUploaderDropzone"] button { border-radius: 10px; }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; font-weight: 600; padding: 0.55rem 1rem; border: 1px solid var(--line);
        background: var(--surface); color: var(--ink) !important;
        box-shadow: 0 2px 6px rgba(15,23,42,0.05); transition: all .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #a5b4fc; transform: translateY(-1px); box-shadow: 0 8px 18px rgba(79,70,229,0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed); border: none; color: #ffffff !important;
        box-shadow: 0 8px 20px rgba(99,102,241,0.38);
    }
    .stButton > button[kind="primary"] * { color: #ffffff !important; }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 12px 26px rgba(99,102,241,0.5); }
    .stButton > button:disabled { opacity: .45; transform: none; box-shadow: none; }

    [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
    button[data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--brand) !important; background: rgba(79,70,229,0.07); }
    [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #4f46e5, #7c3aed) !important; height: 3px !important; }

    [data-testid="stMetric"], [data-testid="metric-container"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 12px 16px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.04);
    }
    [data-testid="stMetricValue"] { font-weight: 700; }
    [data-testid="stAlert"] { border-radius: 14px; }
    textarea { border-radius: 12px !important; }
    [data-testid="stExpander"] { border-radius: 14px; border: 1px solid var(--line); background: var(--surface); }
    [data-testid="stImage"] img { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="brand">
  <div class="mark">OCR</div>
  <div><div class="name">OCR Platform</div><div class="tag">Documents · Local &amp; Online AI</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Settings")

    key_col, btn_col = st.columns([8, 1])
    with key_col:
        typed_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key - never stored. Press the button beside it to send the key and "
                 "load the online models.",
            placeholder="sk-..."
        ).strip()
    with btn_col:
        send_key = st.button("➤", key="send_api_key", help="Send the key and load the online models",
                             use_container_width=True)

    if send_key:
        st.session_state.pop("api_key_ok", None)
        if not typed_key:
            st.warning("Enter your OpenAI API key first.")
        else:
            try:
                _list_account_models(hashlib.sha256(typed_key.encode()).hexdigest()[:12], typed_key)
                st.session_state["api_key_ok"] = typed_key
            except openai.AuthenticationError:
                st.error("This API key was rejected by OpenAI. Check it and try again.")
            except Exception as _e:                      # network / rate limit: the key itself may be fine
                st.session_state["api_key_ok"] = typed_key
                st.caption(f"Could not load your model list ({type(_e).__name__}); showing the default models.")

    # Only a key that was sent (and still matches the box) switches the online side on.
    api_key = typed_key if typed_key and st.session_state.get("api_key_ok") == typed_key else ""

    if api_key:
        _online_options = _online_model_ids(api_key) + ["Custom model ID..."]
        _online_choice = st.selectbox(
            "Online model",
            options=_online_options,
            index=_online_options.index(REFERENCE_MODEL) if REFERENCE_MODEL in _online_options else 0,
            help="OpenAI vision model used for the VLM side. The list comes from your account.",
        )
        if _online_choice == "Custom model ID...":
            vlm_model = st.text_input(
                "Model ID",
                value="",
                placeholder="exact OpenAI model id",
                help="Type the exact API model id of any OpenAI vision-capable model.",
            ).strip()
        else:
            vlm_model = _online_choice
    else:
        vlm_model = ""
        if typed_key:
            st.caption("Press ➤ to send the key and choose an online model.")

    st.markdown("---")
    st.markdown("### Models")

    ocr_engine = st.selectbox(
        "Local model",
        options=[
            "Tesseract OCR",
            "Easy OCR",
            "Paddle OCR",
            "ABG OCR",
            "PaddleOCR-VL",
        ],
        index=0,
        help="Runs on this computer - no internet needed.",
    )

    doc_type = st.selectbox(
        "Document type",
        options=["General", "Egyptian ID card", "Cheque"],
        index=0,
        help="Decides which key fields are shown in the Table tab (and asked from the online model). "
             "ID card: name, birth date, address, 14-digit national number. "
             "Cheque: customer name, amount, cheque date, bank name. "
             "Works with every local model (ABG OCR reads fields from the image and is the most accurate).",
    )

    st.markdown("---")

    # Language selection depends on engine
    ocr_lang = None
    easyocr_langs = None
    paddle_langs = None
    local_langs = None
    id_mode = doc_type == "Egyptian ID card"
    cheque_mode = doc_type == "Cheque"
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
    elif ocr_engine == "PaddleOCR-VL":
        st.caption("Experimental. 0.9B vision-language page parser. On this CPU it needs ~2.5 min "
                   "per page and ~4.5 GB of free RAM. In our Arabic cheque test it misread Arabic "
                   "words and skipped a line, so verify its output; ABG OCR was more accurate.")
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
    st.markdown("""
<div class="side-foot"><b>ABG · Accord Business Group</b><span>Local → Cloud pipeline</span></div>
""", unsafe_allow_html=True)

# Fixed defaults
dpi = 300
max_pages = 3
vlm_prompt = _summary_mod().vlm_prompt(doc_type)

# ─── Header ──────────────────────────────────────────────────────────────────────
engine_label = ocr_engine
st.markdown(f"""
<div class="app-header">
    <h1>OCR Platform</h1>
    <p>Upload a PDF or image, read it with a local model and an online vision model, and compare the results side by side.</p>
    <div class="chips">
        <span class="chip">Local · {engine_label}</span>
        <span class="chip">Online · {vlm_model or "—"}</span>
        <span class="chip">Document · {doc_type}</span>
    </div>
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
        "template_fields": True,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,
        "engine": f"ABG OCR · Egyptian ID · {engine[-2:].upper()} ({', '.join(lang_codes)})",
    }


def run_paddleocr_vl(images: list, id_mode: bool = False) -> dict:
    """PaddleOCR-VL (experimental, slow): one-shot subprocess in the isolated Paddle-3 environment.

    Egyptian ID card (``id_mode``): each image is first straightened to the card (same step as ABG OCR), the
    14-digit number is read by a small glyph classifier (PaddleOCR-VL misreads this digit font), and the VL
    text supplies the name / address / serial. Other document types: the plain VL text, as before."""
    start = time.time()
    import sys
    if LOCAL_DOC_AI_DIR not in sys.path:
        sys.path.insert(0, LOCAL_DOC_AI_DIR)
    from app.ocr.paddle_vl import parse_images, blocks_to_text
    if not id_mode:
        result = parse_images(images)
        full_text = blocks_to_text(result["pages"])
        return {
            "text": full_text,
            "pages": len(images),
            "word_count": len(full_text.split()),
            "char_count": len(full_text),
            "time_sec": round(time.time() - start, 2),
            "cost_usd": 0.0,
            "engine": result.get("model", "PaddleOCR-VL"),
        }

    from app.templates.card import extract_card
    from app.templates.id_card_vl import build_id_fields
    from app.templates.id_number_reader import read_number_fast
    cards = [extract_card(img) for img in images]
    numbers = [read_number_fast(c) for c in cards]
    result = parse_images(cards)
    page_texts = [blocks_to_text([p]) for p in result["pages"]]
    fields = [build_id_fields(t, n) for t, n in zip(page_texts, numbers)]
    full_text = (chr(10) * 2 + "--- Page Break ---" + chr(10) * 2).join(page_texts)
    return {
        "text": full_text,
        "structured": fields if len(fields) > 1 else fields[0],
        "template_fields": True,
        "pages": len(images),
        "word_count": len(full_text.split()),
        "char_count": len(full_text),
        "time_sec": round(time.time() - start, 2),
        "cost_usd": 0.0,
        "engine": result.get("model", "PaddleOCR-VL") + " · straightened card + number reader",
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
        "template_fields": True,
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
    if engine == "PaddleOCR-VL":
        return run_paddleocr_vl(images, id_mode)
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


# USD per 1M tokens: (input, cached input, output) - OpenAI list prices, standard tier.
# Unknown models are still usable, the cost just shows n/a.
MODEL_PRICES = {
    "gpt-6-astra":  (10.00, 1.000, 50.00),
    "gpt-6.1-sol":  (2.00, 0.100, 10.00),
    "gpt-6-sol":    (2.00, 0.200, 10.00),
    "gpt-6-luna":   (0.10, 0.010, 0.50),
    "gpt-4o":       (2.50, 1.250, 10.00),
    "gpt-4o-mini":  (0.15, 0.075, 0.60),
    "o4-mini":      (1.10, 0.275, 4.40),
}


def price_for(model: str):
    """Price tuple for a model id; dated snapshots (gpt-4o-2024-08-06) use their base model's price."""
    m = normalize_model_id(model)
    if m in MODEL_PRICES:
        return MODEL_PRICES[m]
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", m)
    return MODEL_PRICES.get(base)


def compute_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int):
    """(usd, known). Cached input tokens are billed at the cheaper cached rate; reasoning tokens are
    already part of ``output_tokens`` (OpenAI bills them as output)."""
    price = price_for(model)
    if not price:
        return 0.0, False
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * price[0] + cached_tokens * price[1] + output_tokens * price[2]) / 1_000_000, True


def _chat_json(client, model: str, messages: list):
    """Chat call that works across model families.

    Newer / reasoning models (o-series, gpt-5+) reject ``max_tokens`` and spend part of the
    budget on hidden reasoning, so use ``max_completion_tokens`` with a larger budget for them.
    Falls back to the legacy parameters / plain text if a model rejects them."""
    reasoning = model.startswith(("o1", "o3", "o4")) or "gpt-5" in model or "gpt-6" in model
    limit = 16000 if reasoning else 4000
    base = dict(model=model, messages=messages)
    attempts = [
        dict(response_format={"type": "json_object"}, max_completion_tokens=limit),
        dict(response_format={"type": "json_object"}, max_tokens=limit),
        dict(max_completion_tokens=limit),
    ]
    last = None
    for extra in attempts:
        try:
            return client.chat.completions.create(**base, **extra)
        except Exception as e:                       # try the next parameter style
            last = e
            msg = str(e).lower()
            if not any(k in msg for k in ("max_tokens", "max_completion_tokens", "response_format",
                                          "unsupported", "unknown parameter", "json")):
                raise
    raise last


def run_vlm(images: list, api_key: str, model: str, prompt: str) -> dict:
    import httpx
    model = normalize_model_id(model)
    client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
    start = time.time()
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0

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
            resp = _chat_json(client, model, messages)
            content = resp.choices[0].message.content or ""
            total_input_tokens += resp.usage.prompt_tokens
            total_output_tokens += resp.usage.completion_tokens
            total_cached_tokens += getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {"raw_response": content, "parse_error": True}
            parsed["_page"] = i + 1
            all_results.append(parsed)
        except Exception as e:
            err = str(e)
            if "model_not_found" in err or "does not exist" in err:
                err += (f"\n-> '{model}' is not available to this API key. Pick a model from the "
                        "list (it shows what your account can use) or type an exact id.")
            all_results.append({"_page": i + 1, "error": err})

    elapsed = time.time() - start
    cost, cost_known = compute_cost(model, total_input_tokens, total_cached_tokens, total_output_tokens)

    combined_text = ""
    for r in all_results:
        combined_text += r.get("text", r.get("raw_response", "")) + "\n\n"

    return {
        "results": all_results,
        "combined_text": combined_text.strip(),
        "pages": len(images),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cached_tokens": total_cached_tokens,
        "cost_usd": round(cost, 6),
        "cost_known": cost_known,
        "time_sec": round(elapsed, 2),
        "model": model,
    }


def fmt_cost(res: dict) -> str:
    if not res.get("cost_known", True):
        return "n/a"
    c = res["cost_usd"]
    return f"${c:.4f}" if c >= 0.001 or c == 0 else f"${c:.6f}"


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def build_comparison_table(ocr_result: dict, vlm_result: dict) -> dict:
    eng = ocr_result.get("engine", "OCR")
    arabic_note = "✅ Native Arabic" if ("Easy OCR" in eng or "Paddle OCR" in eng or "ABG OCR" in eng or "PaddleOCR-VL" in eng) else "⚠️ Requires ara pack"
    return {
        "Speed (sec)": {
            "OCR": ocr_result["time_sec"],
            "VLM": vlm_result["time_sec"],
            "Winner": "OCR" if ocr_result["time_sec"] < vlm_result["time_sec"] else "VLM"
        },
        "Cost (USD)": {
            "OCR": f"${ocr_result['cost_usd']:.4f}",
            "VLM": fmt_cost(vlm_result),
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


def _entry_row(entry: dict, doc: str):
    """Key-field row of one saved run for document type ``doc`` (None if it cannot be derived)."""
    summary, res = _summary_mod(), entry["result"]
    if entry["kind"] == "ocr":
        if res.get("template_fields") and entry["doc_type"] == doc:
            st_ = res["structured"]
            items = st_ if isinstance(st_, list) else [st_]
        else:                                      # any other local model: parse its plain text
            items = _summary_mod_text().fields_from_text(doc, res["text"])
        out = summary.template_summary(doc, items[0]) if items else None
        return out[0] if out else None
    if entry["doc_type"] != doc:                   # the online answer was asked for other fields
        return None
    pages = [p for p in res["results"] if "error" not in p]
    return summary.vlm_summary(doc, pages[0])[0] if pages else None




def _is_reference_label(label: str) -> bool:
    return normalize_model_id(label) == REFERENCE_MODEL


def _entry_text(entry: dict) -> str:
    """All text a saved run produced (used to compare General documents)."""
    res = entry["result"]
    if entry["kind"] == "ocr":
        return res.get("text", "")
    parts = [res.get("combined_text", "")]
    for p in res.get("results", []):
        if "error" not in p:
            parts += [v for _, v in _summary_mod().flatten_fields({k: x for k, x in p.items() if k != "text"})]
    return " ".join(parts)


def show_compare(hist: dict, current_fhash=None) -> None:
    """Field-by-field comparison of the saved runs of one document against the reference model (REFERENCE_MODEL)."""
    from importlib import import_module
    _summary_mod()
    ref_mod = import_module("app.templates.compare_ref")
    entries = list(hist.values())
    docs = {}
    for e in entries:
        docs.setdefault(e.get("fhash", e["file"]), e["file"])
    order = sorted(docs, key=lambda k: k != current_fhash)             # current document first
    key = order[0]
    if len(order) > 1:
        key = st.selectbox("Document", order, format_func=lambda k: docs[k], key="cmp_doc")
    doc_choices = ["Egyptian ID card", "Cheque", "General"]
    doc = st.radio("Compare fields of", doc_choices, horizontal=True, key="cmp_doc_type",
                   index=doc_choices.index(doc_type) if doc_type in doc_choices else 2)
    general = doc == "General"

    # --- the reference: an online run of this document with the same document type
    cands = {}
    for idx, e in enumerate(entries):
        if e.get("fhash", e["file"]) != key or e["kind"] != "vlm" or e.get("doc_type", "General") != doc:
            continue
        if any("error" not in p for p in e["result"].get("results", [])):
            cands[f"{e['label']}  ({e['at']})"] = e
    ref_names = [n for n, e in cands.items() if _is_reference_label(e["label"])]
    if not ref_names:
        st.warning(f"No reference yet. Run the online model **{REFERENCE_MODEL}** on this document with document type "
                   f"**{doc}** (sidebar), then come back - every other model is scored against it.")
        if cands:
            st.caption("Online runs available as a reference: " + ", ".join(cands))
        return
    ref_name = st.selectbox(f"Reference: {REFERENCE_MODEL} (treated as the correct answer)", list(cands),
                            index=list(cands).index(ref_names[-1]), key="cmp_ref")
    ref_entry = cands[ref_name]
    ref_page = next(p for p in ref_entry["result"]["results"] if "error" not in p)
    ref_row = None if general else _summary_mod().vlm_summary(doc, ref_page)[0]
    fields = ref_mod.reference_fields(doc, ref_row, ref_page)
    if not any(str(v).strip() for _, _, v in fields):
        st.warning("The reference returned no values for this document type, so there is nothing to compare against.")
        return

    # --- the models to score (everything except the reference run itself)
    rows_in, skipped, seen = [], [], {}
    for e in entries:
        if e is ref_entry or e.get("fhash", e["file"]) != key:
            continue
        name = ("Local · " if e["kind"] == "ocr" else "Online · ") + e["label"]
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name += f" #{seen[name]}"
        item = {"label": name, "time": e["result"]["time_sec"]}
        if general:
            item["text"] = _entry_text(e)
        else:
            row = _entry_row(e, doc)
            if row is None:
                skipped.append(name)
                continue
            item["row"] = row
        rows_in.append(item)
    if not rows_in:
        st.info("Nothing to score yet - run another model on this document (with this document type for online models).")
        return
    ref_label = "Reference · " + ref_entry["label"]
    headers, rows, avgs, best, notes = ref_mod.compare_to_reference(doc, fields, rows_in, ref_label)
    st.markdown(_summary_mod().html_table(headers, rows, max_height=600, highlight_headers=[ref_label]),
                unsafe_allow_html=True)
    for n in notes:
        st.warning(n)
    if len(rows_in) > 1:
        top = [rows_in[i]["label"] for i, sc in enumerate(avgs) if sc == avgs[best]]
        st.success(f"Closest to the reference: **{rows_in[best]['label']}** ({avgs[best]:.0f}% average)"
                   + (f" - tied with {', '.join(t for t in top if t != rows_in[best]['label'])}; the faster one is shown."
                      if len(top) > 1 else ""))
    else:
        st.info(f"{rows_in[0]['label']}: {avgs[0]:.0f}% average against the reference. Run another model to compare.")
    if skipped:
        st.caption("Not comparable (online answer was requested for a different document type): " + ", ".join(skipped))
    st.caption("Each cell: found? · correctness vs the reference. Names / addresses / text = share of the "
               "reference's words that the model also wrote; national ID = all 14 digits must match; "
               "amount and date = same value. The reference is not scored. The score is an average of the "
               "fields the reference has. First page of each document is compared.")


def show_history(current_fhash=None) -> None:
    """Every run made in this session (kept until cleared) + a status comparison of the runs."""
    hist = st.session_state.get("history", {})
    if not hist:
        return
    st.markdown("---")
    tab_hist, tab_cmp = st.tabs([f"🕘 History ({len(hist)})", "⚖️ Comparison"])
    with tab_cmp:
        show_compare(hist, current_fhash)
    with tab_hist:
        if st.button("🗑️ Clear all", key="clear_history"):
            st.session_state.history = {}
            st.rerun()
        for i, (hkey, entry) in enumerate(reversed(list(hist.items()))):
            res = entry["result"]
            tag = "LOCAL" if entry["kind"] == "ocr" else "ONLINE"
            title = f"{tag} · {entry['label']} · {entry['doc_type']} · {entry['file']} · {res['time_sec']}s · {entry['at']}"
            c_entry, c_del = st.columns([14, 1])
            with c_del:
                if st.button("🗑️", key=f"del_{abs(hash(str(hkey)))}", help="Delete this item"):
                    hist.pop(hkey, None)
                    st.rerun()
            with c_entry:
                with st.expander(title, expanded=False):
                    res = {**res, "doc_type": entry["doc_type"]}
                    t_table, t_raw = st.tabs(["📊 Table", "🧾 Raw data"])
                    with t_table:
                        (show_ocr_table if entry["kind"] == "ocr" else show_vlm_table)(res)
                    with t_raw:
                        text = res["text"] if entry["kind"] == "ocr" else res["combined_text"]
                        st.text_area("Text", value=text, height=240, key=f"hist_text_{i}")


def _table(df: "pd.DataFrame") -> None:
    """Render a DataFrame (columns given left-to-right) as an RTL table with every value
    right-aligned. st.dataframe cannot align cells, so this is HTML (values are escaped)."""
    rows = df.astype(str).values.tolist()
    st.markdown(_summary_mod().html_table([str(c) for c in df.columns], rows), unsafe_allow_html=True)


def _vertical_table(columns: list, rows: list) -> None:
    """One field per row; the FIELD column is on the right (read right-to-left: field - value)."""
    headers, data = _summary_mod().vertical_table(columns, rows)
    _table(pd.DataFrame(data, columns=headers))


def _pairs_table(pairs: list) -> None:
    headers, data = _summary_mod().pairs_table(pairs)
    _table(pd.DataFrame(data, columns=headers))


def _summary_mod_text():
    _summary_mod()                                # puts local-document-ai on sys.path
    from app.templates import text_fields
    return text_fields


def _warnings(items: list) -> None:
    for w in dict.fromkeys(items):              # unique, order kept
        st.warning(w)


def show_ocr_table(res: dict) -> None:
    """Key fields of the local model result (ID card / cheque via ABG OCR), else the lines read."""
    summary = _summary_mod()
    doc = res.get("doc_type", "General")
    structured = res.get("structured")
    from_text = False
    if doc != "General" and not (structured and res.get("template_fields")):
        structured = _summary_mod_text().fields_from_text(doc, res["text"])      # any engine: parse its text
        from_text = True
    if doc != "General" and structured:
        items = structured if isinstance(structured, list) else [structured]
        rows, warns = [], []
        for it in items:
            out = summary.template_summary(doc, it)
            if out:
                rows.append(out[0])
                warns += out[1]
        if rows:
            _vertical_table(summary.columns_for(doc), rows)
            if from_text:
                warns.append("Fields were parsed from this model's plain text; ABG OCR reads them from the image and is more accurate.")
            _warnings(warns)
            return
    lines = [ln for ln in res["text"].splitlines() if ln.strip() and not ln.startswith("--- Page Break")]
    _table(pd.DataFrame({"Text": lines, "#": range(1, len(lines) + 1)}))


def show_vlm_table(res: dict) -> None:
    """Key fields the online model returned for the chosen document type."""
    summary = _summary_mod()
    doc = res.get("doc_type", "General")
    pages = [p for p in res["results"] if "error" not in p]
    for p in res["results"]:
        if "error" in p:
            st.error(p["error"])
    if not pages:
        return                                   # only errors: no empty table under them
    if doc != "General":
        rows, warns = [], []
        for p in pages:
            row, w = summary.vlm_summary(doc, p)
            rows.append(row)
            warns += w
        _vertical_table(summary.columns_for(doc), rows)
        _warnings(warns)
        return
    rows = []
    for p in pages:
        if isinstance(p.get("fields"), dict):
            rows += summary.flatten_fields(p["fields"])
    if rows:
        _pairs_table(rows)
    else:
        lines = [ln for ln in res["combined_text"].splitlines() if ln.strip()]
        _table(pd.DataFrame({"Text": lines, "#": range(1, len(lines) + 1)}))


def show_fields_table(obj) -> None:
    """Every field of a structured result (raw-data view)."""
    rows = _summary_mod().flatten_fields(obj)
    if rows:
        st.markdown("**All extracted fields**")
        _pairs_table(rows)


# ─── Main Logic ──────────────────────────────────────────────────────────────────
if uploaded_file is None:
    show_history()
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
                                    disabled=not api_key or not vlm_model)
    run_both = col_btn3.button("🔬 Run BOTH & Compare", use_container_width=True, type="primary",
                                disabled=not api_key or not vlm_model)

    if not api_key and (run_vlm_only or run_both):
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to use VLM.")

    # Every run is kept in session history, keyed by (document, model, settings): picking another
    # model never discards earlier results, and re-running the same thing reuses the saved result.
    import hashlib
    hist = st.session_state.setdefault("history", {})
    fhash = hashlib.md5(file_bytes).hexdigest()
    _abg = ocr_engine in ("ABG OCR", "PaddleOCR-VL")      # results depend on the document type
    ocr_key = ("ocr", fhash, ocr_engine, doc_type if _abg else "-", str(ocr_lang), str(easyocr_langs),
               str(paddle_langs), fix_rtl, str(local_langs), local_engine, enhance_image)
    vlm_key = ("vlm", fhash, vlm_model, doc_type)

    def _save(key, kind, label, result):
        result["doc_type"] = doc_type
        hist[key] = {"kind": kind, "label": label, "doc_type": doc_type, "file": uploaded_file.name,
                     "result": result, "at": time.strftime("%H:%M:%S"), "fhash": fhash}

    if (run_ocr or run_both) and ocr_key in hist:
        st.info(f"♻️ {ocr_engine}: using the saved result for this document (not re-run). "
                "Clear the history to run it again.")
    elif run_ocr or run_both:
        spinner_msg = f"⚡ Running {ocr_engine}... first run loads the models, please wait"
        with st.spinner(spinner_msg):
            try:
                own_preprocessing = ocr_engine in ("ABG OCR", "PaddleOCR-VL")
                ocr_images = (
                    [preprocess_for_ocr(img) for img in images]
                    if enhance_image and not own_preprocessing else images
                )
                r = run_ocr_engine(
                    ocr_images, ocr_engine, ocr_lang, easyocr_langs, paddle_langs, fix_rtl,
                    local_langs, id_mode, local_engine, cheque_mode
                )
                _save(ocr_key, "ocr", ocr_engine, r)
                st.success(f"✅ OCR done in {r['time_sec']}s — {r['engine']}")
            except Exception as e:
                st.error(f"OCR error: {e}")

    if (run_vlm_only or run_both) and api_key and vlm_key in hist:
        st.info(f"♻️ {vlm_model}: using the saved result for this document (not re-run). "
                "Clear the history to run it again.")
    elif run_vlm_only or run_both:
        if api_key:
            with st.spinner(f"🤖 Running {vlm_model} Vision... (~5-20s per page)"):
                try:
                    r = run_vlm(images, api_key, vlm_model, vlm_prompt)
                    _save(vlm_key, "vlm", vlm_model, r)
                    st.success(f"✅ VLM done in {r['time_sec']}s · Cost: {fmt_cost(r)}")
                except Exception as e:
                    st.error(f"VLM error: {e}")

    # Show the saved result of the CURRENTLY selected models for this document.
    ocr_res = hist[ocr_key]["result"] if ocr_key in hist else None
    vlm_res = hist[vlm_key]["result"] if vlm_key in hist else None
    if ocr_res:
        ocr_res = {**ocr_res, "doc_type": hist[ocr_key]["doc_type"]}

    if ocr_res or vlm_res:
        st.markdown("---")

        left, right = st.columns(2)

        with left:
            st.markdown(f"""
<div class="panel-title ocr"><span class="dot"></span>{ocr_engine}<small>LOCAL</small></div>
""", unsafe_allow_html=True)

            if ocr_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Time", f"{ocr_res['time_sec']}s")
                c2.metric("Words", f"{ocr_res['word_count']:,}")
                c3.metric("Cost", "$0.00")
                tab_table, tab_raw = st.tabs(["📊 Table", "🧾 Raw data"])
                with tab_table:
                    show_ocr_table(ocr_res)
                with tab_raw:
                    st.text_area("Raw OCR Text", value=ocr_res["text"], height=360, key="ocr_text_display")
                    st.download_button("⬇️ Download OCR Text", data=ocr_res["text"],
                                       file_name="ocr_output.txt", mime="text/plain")
                    if ocr_res.get("structured"):
                        show_fields_table(ocr_res["structured"])
            else:
                st.info("Run OCR to see results here.")

        with right:
            st.markdown(f"""
<div class="panel-title vlm"><span class="dot"></span>{vlm_res['model'] if vlm_res else vlm_model}<small>ONLINE</small></div>
""", unsafe_allow_html=True)

            if vlm_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Time", f"{vlm_res['time_sec']}s")
                c2.metric("Tokens", f"{vlm_res['input_tokens'] + vlm_res['output_tokens']:,}")
                c3.metric("Cost", fmt_cost(vlm_res))
                tab_table, tab_raw = st.tabs(["📊 Table", "🧾 Raw data"])
                with tab_table:
                    show_vlm_table(vlm_res)
                with tab_raw:
                    st.text_area("Extracted Text (from VLM)", value=vlm_res["combined_text"],
                                 height=360, key="vlm_text_display")
                    st.download_button("⬇️ Download VLM Text", data=vlm_res["combined_text"],
                                       file_name="vlm_output.txt", mime="text/plain")
                    show_fields_table([{k: v for k, v in p.items() if k != "_page"} for p in vlm_res["results"]])
            else:
                st.info("Run VLM to see results here.")

    show_history(fhash)

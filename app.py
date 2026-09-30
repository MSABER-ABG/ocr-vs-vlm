"""
OCR vs VLM Comparison App
Accord Business Group
Compare Tesseract OCR vs GPT-4o Vision on PDF/image files
"""

import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import openai
import base64
import io
import time
import json
import os
import shutil
import tempfile
from pathlib import Path

# ─── Tesseract binary auto-detect (Windows) ──────────────────────────────────────
# pytesseract needs to know where tesseract.exe lives if it isn't on PATH.
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
    /* Main background + force dark text everywhere */
    .stApp { background-color: #f5f7fa; color: #1a1a2e !important; }
    .stApp p, .stApp span, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stSelectbox label, .stSlider label, .stTextInput label,
    .stTextArea label, .stFileUploader label { color: #1a1a2e !important; }

    /* Sidebar */
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

    /* Cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 16px;
        border-left: 4px solid #3b3bbf;
    }
    .result-card.vlm { border-left-color: #10b981; }

    /* Metric boxes */
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

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1a6e 0%, #3b3bbf 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 { color: white !important; margin: 0; font-size: 28px; }
    .app-header p { color: white !important; margin: 4px 0 0; }

    /* Decision badge */
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

    /* Stmetric override */
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

    ocr_lang = st.selectbox(
        "OCR Language",
        options=["ara", "eng", "ara+eng", "osd"],
        index=0,
        help="Tesseract language pack"
    )

    vlm_model = st.selectbox(
        "VLM Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0
    )

    st.markdown("---")
    st.markdown("**Accord Business Group**")
    st.markdown("*Local → Cloud Pipeline*")

# Fixed defaults (hidden from UI)
dpi = 150
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
st.markdown("""
<div class="app-header">
    <h1>🔬 OCR vs VLM — Live Comparison</h1>
    <p>Upload a PDF or image · Run Tesseract OCR &amp; GPT-4o Vision · Compare side-by-side</p>
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
    """Convert PDF bytes → list of PIL Images."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages = pdf2image.convert_from_path(tmp_path, dpi=dpi)
        return pages[:max_pages]
    finally:
        os.unlink(tmp_path)


def image_to_base64(img: Image.Image) -> str:
    """PIL Image → base64 string for API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_tesseract(images: list, lang: str) -> dict:
    """Run Tesseract on list of PIL images. Returns aggregated result."""
    start = time.time()
    texts = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img, lang=lang)
            texts.append(text)
        except Exception as e:
            texts.append(f"[Error on page: {e}]")
    elapsed = time.time() - start
    full_text = "\n\n--- Page Break ---\n\n".join(texts)
    word_count = len(full_text.split())
    char_count = len(full_text)
    return {
        "text": full_text,
        "pages": len(images),
        "word_count": word_count,
        "char_count": char_count,
        "time_sec": round(elapsed, 2),
        "cost_usd": 0.0,   # Tesseract is free
    }


def run_vlm(images: list, api_key: str, model: str, prompt: str) -> dict:
    """Run GPT-4o Vision on list of PIL images. Returns aggregated result."""
    import httpx
    client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
    start = time.time()

    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, img in enumerate(images):
        b64 = image_to_base64(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=4000
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

    # Cost calculation (GPT-4o 2026 pricing)
    if model == "gpt-4o":
        cost = (total_input_tokens / 1_000_000 * 2.50) + (total_output_tokens / 1_000_000 * 10.00)
    else:  # gpt-4o-mini
        cost = (total_input_tokens / 1_000_000 * 0.15) + (total_output_tokens / 1_000_000 * 0.60)

    # Extract combined text for comparison
    combined_text = ""
    for r in all_results:
        if "text" in r:
            combined_text += r["text"] + "\n\n"
        elif "raw_response" in r:
            combined_text += r["raw_response"] + "\n\n"

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
    """Build comparison metrics dict."""
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
            "OCR": "⚠️ Requires ara pack",
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


# ─── Decision Framework ───────────────────────────────────────────────────────────
def show_decision_framework():
    st.markdown("### 🧭 Decision Framework")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="result-card">
<h4>✅ Use OCR (Tesseract) when:</h4>
<ul>
<li>High volume, low cost is priority</li>
<li>Simple printed text (no tables/forms)</li>
<li>Offline / air-gapped environment</li>
<li>Latin scripts with clean scans</li>
<li>Speed is critical (&lt; 1 sec/page)</li>
<li>No API key / internet required</li>
</ul>
<span class="badge-ocr">Best for: Bulk text extraction</span>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="result-card vlm">
<h4>✅ Use VLM (GPT-4o Vision) when:</h4>
<ul>
<li>Arabic, mixed-language, or handwritten</li>
<li>Forms with key-value pairs to extract</li>
<li>Tables, charts, or structured data</li>
<li>Low volume, high accuracy needed</li>
<li>Semantic understanding required</li>
<li>Output must be structured JSON</li>
</ul>
<span class="badge-vlm">Best for: Banking forms, contracts, reports</span>
</div>
""", unsafe_allow_html=True)


# ─── Main Logic ──────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👆 Upload a PDF or image to begin the comparison.")
    st.markdown("---")
    show_decision_framework()

else:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()

    # ── Convert to images ──
    with st.spinner("📄 Preparing file..."):
        if file_ext == ".pdf":
            try:
                images = render_pdf_pages(file_bytes, dpi, max_pages)
                st.success(f"✅ PDF rendered: {len(images)} page(s) at {dpi} DPI")
            except Exception as e:
                st.error(f"PDF render error: {e}\n\nTip: Install poppler → add to PATH")
                st.stop()
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                images = [img]
                st.success(f"✅ Image loaded: {img.size[0]}×{img.size[1]}px")
            except Exception as e:
                st.error(f"Image load error: {e}")
                st.stop()

    # ── Preview ──
    with st.expander("🖼️ Preview pages", expanded=False):
        cols = st.columns(min(len(images), 3))
        for i, (img, col) in enumerate(zip(images, cols)):
            col.image(img, caption=f"Page {i+1}")

    # ── Run buttons ──
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    run_ocr = col_btn1.button("⚡ Run OCR Only", use_container_width=True, type="secondary")
    run_vlm_only = col_btn2.button("🤖 Run VLM Only", use_container_width=True, type="secondary",
                                    disabled=not api_key)
    run_both = col_btn3.button("🔬 Run BOTH & Compare", use_container_width=True, type="primary",
                                disabled=not api_key)

    if not api_key and (run_vlm_only or run_both):
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to use VLM.")

    # ── Session state ──
    if "ocr_result" not in st.session_state:
        st.session_state.ocr_result = None
    if "vlm_result" not in st.session_state:
        st.session_state.vlm_result = None

    # ── Execute OCR ──
    if run_ocr or run_both:
        with st.spinner("⚡ Running Tesseract OCR..."):
            try:
                st.session_state.ocr_result = run_tesseract(images, ocr_lang)
                st.success(f"✅ OCR done in {st.session_state.ocr_result['time_sec']}s")
            except Exception as e:
                st.error(f"OCR error: {e}")
                st.session_state.ocr_result = None

    # ── Execute VLM ──
    if run_vlm_only or run_both:
        if api_key:
            with st.spinner(f"🤖 Running {vlm_model} Vision... (this takes ~5-20s per page)"):
                try:
                    st.session_state.vlm_result = run_vlm(images, api_key, vlm_model, vlm_prompt)
                    st.success(
                        f"✅ VLM done in {st.session_state.vlm_result['time_sec']}s "
                        f"· Cost: ${st.session_state.vlm_result['cost_usd']:.4f}"
                    )
                except Exception as e:
                    st.error(f"VLM error: {e}")
                    st.session_state.vlm_result = None

    # ─── Display Results ──────────────────────────────────────────────────────

    ocr_res = st.session_state.ocr_result
    vlm_res = st.session_state.vlm_result

    if ocr_res or vlm_res:
        st.markdown("---")

        # ── Summary metrics ──
        if ocr_res and vlm_res:
            st.markdown("### 📊 Head-to-Head Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ OCR Speed", f"{ocr_res['time_sec']}s")
            m2.metric("🤖 VLM Speed", f"{vlm_res['time_sec']}s")
            m3.metric("💰 OCR Cost", "$0.0000")
            m4.metric("💰 VLM Cost", f"${vlm_res['cost_usd']:.4f}")

            st.markdown("---")

        # ── Side-by-side ──
        left, right = st.columns(2)

        with left:
            st.markdown("""
<div style="background:#dbeafe;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
<b style="color:#1e40af;font-size:16px;">⚡ OCR — Tesseract</b><br>
<span style="color:#3b82f6;font-size:12px;">Free · Offline · Raw text</span>
</div>
""", unsafe_allow_html=True)

            if ocr_res:
                c1, c2, c3 = st.columns(3)
                c1.metric("Time", f"{ocr_res['time_sec']}s")
                c2.metric("Words", f"{ocr_res['word_count']:,}")
                c3.metric("Cost", "$0.00")

                st.text_area(
                    "Raw OCR Text",
                    value=ocr_res["text"],
                    height=400,
                    key="ocr_text_display"
                )

                # Download
                st.download_button(
                    "⬇️ Download OCR Text",
                    data=ocr_res["text"],
                    file_name="ocr_output.txt",
                    mime="text/plain"
                )
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

                # Tabs: JSON | Text | Raw
                tab_json, tab_text, tab_raw = st.tabs(["📋 JSON", "📝 Text", "🔢 Raw"])

                with tab_json:
                    for page_result in vlm_res["results"]:
                        page_num = page_result.get("_page", "?")
                        with st.expander(f"Page {page_num}", expanded=True):
                            display = {k: v for k, v in page_result.items() if k != "_page"}
                            st.json(display)

                with tab_text:
                    st.text_area(
                        "Extracted Text (from VLM)",
                        value=vlm_res["combined_text"],
                        height=400,
                        key="vlm_text_display"
                    )

                with tab_raw:
                    st.code(
                        json.dumps(vlm_res["results"], ensure_ascii=False, indent=2),
                        language="json"
                    )

                # Download JSON
                json_str = json.dumps(vlm_res["results"], ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Download VLM JSON",
                    data=json_str,
                    file_name="vlm_output.json",
                    mime="application/json"
                )
            else:
                st.info("Run VLM to see results here.")

        # ── Comparison Table ──
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

            # Verdict
            st.markdown("---")
            ocr_wins = sum(1 for v in comp.values() if v["Winner"] == "OCR")
            vlm_wins = sum(1 for v in comp.values() if v["Winner"] == "VLM")

            if vlm_wins > ocr_wins:
                st.success(f"🏆 **VLM wins {vlm_wins}-{ocr_wins}** for this document — recommended for Arabic/structured extraction.")
            elif ocr_wins > vlm_wins:
                st.info(f"🏆 **OCR wins {ocr_wins}-{vlm_wins}** for this document — recommended for speed/cost-sensitive workflows.")
            else:
                st.warning(f"🤝 **Tie {ocr_wins}-{vlm_wins}** — choose based on your priority: cost (OCR) or accuracy (VLM).")

        # ── Decision Framework (always show at bottom) ──
        st.markdown("---")
        show_decision_framework()

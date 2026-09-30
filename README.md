# 🔬 OCR vs VLM — Comparison App

**Accord Business Group**  
Compare Tesseract OCR vs GPT-4o Vision side-by-side on any PDF or image.

---

## Quick Start

### Step 1 — Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | [python.org](https://python.org) |
| Tesseract OCR | See below |
| Poppler (for PDFs) | See below |
| OpenAI API Key | Entered in the app sidebar |

### Step 2 — Install Tesseract

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. For Arabic: during install, check "Additional language data → Arabic"
4. Test: `tesseract --version`

**Mac:**
```bash
brew install tesseract tesseract-lang
```

**Linux/Ubuntu:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-ara
```

### Step 3 — Install Poppler (for PDF support)

**Windows:**
1. Download: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\poppler`
3. Add `C:\poppler\Library\bin` to System PATH
4. Test: `pdftoppm -v`

**Mac:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt install poppler-utils
```

### Step 4 — Setup Python

```bash
cd ocr-vs-vlm
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Step 5 — Run

```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## Features

| Feature | Description |
|---|---|
| **File Upload** | PDF (multi-page) or PNG/JPG/TIFF images |
| **OCR Engine** | Tesseract — free, offline, raw text |
| **VLM Engine** | GPT-4o Vision — structured JSON extraction |
| **Side-by-side** | Results shown in two columns |
| **Comparison Table** | Speed, cost, words, structure, language support |
| **Decision Verdict** | App recommends OCR or VLM for your document |
| **Downloads** | Export OCR text (.txt) and VLM output (.json) |
| **Decision Framework** | When to use each technology |

---

## Sidebar Settings

- **OpenAI API Key** — entered securely (never stored to disk)
- **OCR Language** — `ara`, `eng`, `ara+eng`, etc.
- **VLM Model** — `gpt-4o` or `gpt-4o-mini`
- **DPI** — PDF render quality (72–300)
- **Max Pages** — limit pages to control cost
- **VLM Prompt** — customize what GPT-4o extracts

---

## Cost Reference (2026)

| Operation | Cost |
|---|---|
| Tesseract OCR | **Free** |
| GPT-4o Vision | $2.50/M input + $10/M output |
| GPT-4o-mini Vision | $0.15/M input + $0.60/M output |
| Typical 1-page form | ~$0.003–0.01 with gpt-4o |

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo → select `app.py`
4. Add `OPENAI_API_KEY` in Secrets (optional — users can enter in sidebar)
5. Deploy!

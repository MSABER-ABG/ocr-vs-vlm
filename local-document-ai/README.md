# local-document-ai

Fully local Document AI pipeline. No VLM, no cloud, no external API, no Docker.

```
load (PDF/image) -> preprocess -> OCR engine -> reading order -> JSON + TXT
```

## Run

```bash
pip install -r requirements.txt
python -m app.cli input/scan.pdf -o output --engine onnx_v5      # accurate (PP-OCRv5, CPU)
python -m app.cli input/scan.pdf -o output                        # fast (PP-OCRv4)
python -m app.cli --template egypt-id input/id.jpg -o output      # Egyptian national ID card
```

Cheque reading (fields + cross-checks) is available from Python / the web lab:

```python
from app.config import Config; from app.pipeline import DocumentAI
from app.templates.cheque import ChequeReader
result = ChequeReader(DocumentAI(Config(engine="onnx_v5"))).read(pil_image)
result.fields   # date, payee, amount_digits/value, amount_words(+value), match, issues, needs_review
```

Output per file: `<name>.json` (lines with bbox + confidence + words) and `<name>.txt`.

## Engines (`app/ocr/`)

| `--engine` | Model | Notes |
|---|---|---|
| `onnx_v5` | PP-OCRv5 server detector + Arabic recognizer | **Recommended.** onnxruntime on CPU, ~10 s/page. ONNX models (~100 MB) download once into `models/`. |
| `paddle_v4` | PP-OCRv4 (PaddlePaddle 2.6) | Fast, in-process. Also used for digit-by-digit reading on ID cards. |
| `paddle_v5` | PP-OCRv5 via PaddlePaddle 3 | Needs an isolated `.venv_v5` (worker process). Superseded by `onnx_v5`; kept as a fallback. |

All engines implement `OCREngine.predict()`; add a new one by implementing it and
registering it in `create_engine()`.

## Notes

- Bounding boxes are in ORIGINAL page pixels (preprocessing scale is undone).
- Preprocessing flattens paper/watermark patterns (background normalisation) - big win on cheques.
- Reading order is script-aware: Arabic runs read right-to-left, Latin/number runs left-to-right.
- CTC models output Arabic in visual order; `app/ocr/text.py::visual_to_logical` restores logical order.
- ID card template: card detection + perspective fix, fixed header text, digit-by-digit national
  number reading, and structural validation (length, birth date, governorate). Failed validation
  sets `needs_review` instead of inventing values.

- Cheque template: finds each field by content + position (date, currency/amount, amount in words
  in English or Arabic over several lines, payee Arabic or Latin), snaps the fixed printed Arabic
  labels to a closed vocabulary, and cross-checks words vs digits. Tilted photos are also tried
  straightened; the result that passes more checks wins. Failed checks set `needs_review`.
  The vocabularies in `app/templates/cheque.py` are small seed lists - extend them per bank.

## Tests

```bash
python -m tests.test_text_order
python -m tests.test_reading_order
python -m tests.test_egypt_id
python -m tests.smoke_arabic     # v4 engine
python -m tests.smoke_onnx       # v5 ONNX engine
python -m tests.test_cheque      # cheque parsing + extraction (synthetic boxes)
python -m tests.regress_cheques  # the 4 real cheque photos kept in input/
```

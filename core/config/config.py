"""
Central config — override any value via environment variables.

Usage in code:
    from core.config import PLATE_MODEL_PATH, OCR_MODEL_PATH, PLATE_CONF, OCR_CONF, DEVICE
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).resolve().parent.parent   # iranian-alpr-api/

PLATE_MODEL_PATH = Path(os.getenv(
    "PLATE_MODEL_PATH",
    str(_BASE / "core" / "models" / "plate_detector.pt")
))

OCR_MODEL_PATH = Path(os.getenv(
    "OCR_MODEL_PATH",
    str(_BASE / "core" / "models" / "ocr_detector.pt")
))

# ── Thresholds ────────────────────────────────────────────────────────────────
PLATE_CONF = float(os.getenv("PLATE_CONF", "0.4"))
OCR_CONF   = float(os.getenv("OCR_CONF",   "0.25"))

# ── Runtime ───────────────────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cpu")

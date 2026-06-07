# -*- coding: utf-8 -*-
"""
config_registry.py — selects detector and OCR implementation at runtime.

Backends are chosen via environment variables:
    DETECTOR_TYPE = "yolo" | "vendor"   (default: "yolo")
    OCR_TYPE      = "yolo" | "vendor"   (default: "yolo")

Adding a new backend:
  1. Implement the BaseDetector or BaseOCR interface.
  2. Add the dotted class path to MODEL_CONFIG below.
  3. Set the environment variable to your new key.
"""

import os
import importlib
from typing import Type

# ── Backend selection ─────────────────────────────────────────────────────────
DETECTOR_TYPE: str = os.getenv("DETECTOR_TYPE", "yolo")
OCR_TYPE: str      = os.getenv("OCR_TYPE",      "yolo")

# ── Registry ──────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "detector": {
        "yolo":   "core.detector.plate_detector.YOLOPlateDetector",
        "vendor": "core.detector.vendor_detector.VendorDetector",
    },
    "ocr": {
        "yolo":   "core.ocr.yolo_ocr.YOLOOCR",
        "vendor": "core.ocr.vender_ocr.VendorOCR",   # note: file is vender_ocr.py (typo in original)
    },
}


# ── Loader ────────────────────────────────────────────────────────────────────

def _load_class(dotted_path: str) -> Type:
    """
    Dynamically import and return a class from a dotted module path.
    e.g. "core.detector.plate_detector.YOLOPlateDetector"
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Could not import module '{module_path}'. "
            f"Check that the file exists and all dependencies are installed."
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_path}' has no class '{class_name}'."
        ) from exc


def get_detector_class() -> Type:
    """Return the detector class selected by DETECTOR_TYPE."""
    if DETECTOR_TYPE not in MODEL_CONFIG["detector"]:
        raise ValueError(
            f"Unknown DETECTOR_TYPE '{DETECTOR_TYPE}'. "
            f"Valid options: {list(MODEL_CONFIG['detector'].keys())}"
        )
    return _load_class(MODEL_CONFIG["detector"][DETECTOR_TYPE])


def get_ocr_class() -> Type:
    """Return the OCR class selected by OCR_TYPE."""
    if OCR_TYPE not in MODEL_CONFIG["ocr"]:
        raise ValueError(
            f"Unknown OCR_TYPE '{OCR_TYPE}'. "
            f"Valid options: {list(MODEL_CONFIG['ocr'].keys())}"
        )
    return _load_class(MODEL_CONFIG["ocr"][OCR_TYPE])

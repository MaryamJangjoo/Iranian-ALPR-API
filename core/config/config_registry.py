"""
config_registry.py — selects detector and OCR implementation
"""

import os
import importlib

DETECTOR_TYPE = os.getenv("DETECTOR_TYPE", "yolo")
OCR_TYPE = os.getenv("OCR_TYPE", "yolo")

MODEL_CONFIG = {
    "detector": {
        "yolo": "core.detector.plate_detector.YOLOPlateDetector",
        "vendor": "core.detector.vendor_detector.VendorDetector",
    },
    "ocr": {
        "yolo": "core.ocr.yolo_ocr.YOLOOCR",
        "vendor": "core.ocr.vendor_ocr.VendorOCR",
    },
}


def _load_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_detector_class():
    return _load_class(MODEL_CONFIG["detector"][DETECTOR_TYPE])


def get_ocr_class():
    return _load_class(MODEL_CONFIG["ocr"][OCR_TYPE])

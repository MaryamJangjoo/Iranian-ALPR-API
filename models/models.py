from ultralytics import YOLO
from core.config import PLATE_MODEL_PATH, OCR_MODEL_PATH

_plate_model = None
_ocr_model = None


def get_plate_model():
    global _plate_model

    if _plate_model is None:
        print("[INFO] Loading Plate Detector...")
        _plate_model = YOLO(str(PLATE_MODEL_PATH))

    return _plate_model


def get_ocr_model():
    global _ocr_model

    if _ocr_model is None:
        print("[INFO] Loading OCR Detector...")
        _ocr_model = YOLO(str(OCR_MODEL_PATH))

    return _ocr_model

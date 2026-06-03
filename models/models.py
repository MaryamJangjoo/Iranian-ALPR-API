from ultralytics import YOLO
from core.config import (
    PLATE_MODEL_PATH,
    OCR_MODEL_PATH,
    DEVICE
)

print("[INFO] Loading Plate Detector...")
plate_model = YOLO(str(PLATE_MODEL_PATH))

print("[INFO] Loading OCR Detector...")
ocr_model = YOLO(str(OCR_MODEL_PATH))

print("[INFO] Models Loaded Successfully")


def get_plate_model():
    return plate_model


def get_ocr_model():
    return ocr_model

# -*- coding: utf-8 -*-
"""
yolo_ocr.py — YOLOv8-based character OCR, Stage 2 of the ALPR pipeline.

Fix applied vs original:
  - Confidence threshold now read from core.config.OCR_CONF instead of
    being hardcoded as 0.1 in model.predict() and a second filter at 0.25.
  - Single, consistent threshold applied at the model.predict() level.
  - iou parameter kept tunable via OCR_IOU env var (default 0.3).
"""

import os
import logging
from typing import List, Dict

from ultralytics import YOLO

from core.interfaces.ocr import BaseOCR
from core.config.config import OCR_MODEL_PATH, OCR_CONF, DEVICE

logger = logging.getLogger(__name__)

# IOU threshold for NMS inside YOLO OCR (can be overridden via env var)
OCR_IOU: float = float(os.getenv("OCR_IOU", "0.3"))


class YOLOOCR(BaseOCR):
    """
    Runs the OCR YOLOv8 model on a cropped plate image and returns
    a list of character detections sorted left-to-right.
    """

    def __init__(self) -> None:
        if not OCR_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"OCR model not found: {OCR_MODEL_PATH}\n"
                "Place the trained ocr_detector.pt in core/models/."
            )
        self.model = YOLO(str(OCR_MODEL_PATH))
        logger.info(
            "YOLOOCR loaded: model=%s  conf=%.2f  iou=%.2f  device=%s",
            OCR_MODEL_PATH.name, OCR_CONF, OCR_IOU, DEVICE,
        )

    def recognize(self, image) -> List[Dict]:
        """
        Run OCR on a cropped plate image.

        Args:
            image: BGR np.ndarray — the normalised plate crop.

        Returns:
            List of dicts, sorted by x1 (left to right):
            [
                {
                    "class_id":   int,
                    "confidence": float,
                    "bbox":       [x1, y1, x2, y2],
                }
            ]
        """
        if image is None:
            return []

        results = self.model.predict(
            source=image,
            conf=OCR_CONF,    # single consistent threshold from config / env
            iou=OCR_IOU,
            device=DEVICE,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        detections: List[Dict] = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            detections.append({
                "class_id":   cls_id,
                "confidence": conf,
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
            })

        # Sort left-to-right by x1
        detections.sort(key=lambda d: d["bbox"][0])

        logger.debug("OCR: %d chars detected (conf≥%.2f)", len(detections), OCR_CONF)
        return detections


def get_ocr() -> YOLOOCR:
    """Factory helper — returns a ready-to-use YOLOOCR instance."""
    return YOLOOCR()

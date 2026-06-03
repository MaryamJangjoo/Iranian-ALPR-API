"""
ALPRPipeline â€” detector â†’ cropper â†’ OCR â†’ reconstruction â†’ validation
"""

from typing import Callable, Dict, Any
import numpy as np
import logging

from core.ocr.reconstruct import (
    reconstruct,
    format_plate,
    is_valid_plate
)

logging.basicConfig(level=logging.INFO)


class ALPRPipeline:

    def __init__(
        self,
        detector,
        ocr,
        cropper: Callable,
        reconstruct: Callable = None,
    ):
        self.detector = detector
        self.ocr = ocr
        self.cropper = cropper

    def run(self, frame: np.ndarray) -> Dict[str, Any]:

        # -----------------------
        # 1. DETECTION
        # -----------------------
        detections = self.detector.detect(frame)
        if not detections:
            return {
                "plate_raw": None,
                "plate": None,
                "valid": False,
                "confidence": 0.0,
                "ocr_chars": 0
            }

        best = max(detections, key=lambda d: d["confidence"])

        # -----------------------
        # 2. CROP
        # -----------------------
        plate_img = self.cropper(frame, best["bbox"])
        if plate_img is None:
            return {
                "plate_raw": None,
                "plate": None,
                "valid": False,
                "confidence": best["confidence"],
                "ocr_chars": 0
            }

        # -----------------------
        # 3. OCR
        # -----------------------
        char_detections = self.ocr.recognize(plate_img)

        # -----------------------
        # 4. RECONSTRUCTION
        # -----------------------
        raw_plate = reconstruct(char_detections)

        formatted_plate = format_plate(raw_plate)
        valid = is_valid_plate(formatted_plate)

        # -----------------------
        # 5. FINAL RESPONSE
        # -----------------------
        return {
            "plate_raw": raw_plate,
            "plate": formatted_plate,
            "valid": valid,
            "confidence": round(best["confidence"], 3),
            "ocr_chars": len(char_detections)
        }

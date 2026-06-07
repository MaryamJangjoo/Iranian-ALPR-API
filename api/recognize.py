# -*- coding: utf-8 -*-
"""
recognize.py — response serialization helpers for the /recognize endpoint.

Fix applied: validate_plate renamed to is_valid_plate to match reconstruct.py export.
"""

from core.ocr.reconstruct import reconstruct, format_plate, is_valid_plate


def plate_handler(detections: list) -> dict:
    """
    Convert a list of YOLOOCR detections into the standard API response dict.

    Args:
        detections: list of dicts from YOLOOCR.recognize()
                    Each dict: {"class_id": int, "confidence": float, "bbox": [...]}

    Returns:
        {
            "plate_raw": str,   raw concatenated characters, e.g. "12ب34567"
            "plate":     str,   formatted,  e.g. "12 ب 345 ایران 67"
            "valid":     bool,  True if plate passes regex validation
        }
    """
    raw       = reconstruct(detections)
    formatted = format_plate(raw)

    return {
        "plate_raw": raw,
        "plate":     formatted,
        "valid":     is_valid_plate(formatted),
    }

# -*- coding: utf-8 -*-
"""
reconstruct.py — plate reconstruction for the API service (Iranian-ALPR-API repo)

CLASS_MAP is 0-indexed (0–9 = digits, 10–41 = Persian letters).
This matches the OCR-only YOLO model output where the plate bounding-box class
has already been removed (see scripts/remove_plate_class.py in Iranian_ALPR).
"""
from __future__ import annotations

import re
from typing import List, Dict, Any

# ------------------------------------------------------------------
# CLASS MAP  (0-indexed, OCR model only — no plate class)
# ------------------------------------------------------------------
CLASS_MAP: Dict[int, str] = {
    0:  "0",
    1:  "1",
    2:  "2",
    3:  "3",
    4:  "4",
    5:  "5",
    6:  "6",
    7:  "7",
    8:  "8",
    9:  "9",
    10: "ا",
    11: "ب",
    12: "پ",
    13: "ت",
    14: "ث",
    15: "ج",
    16: "چ",
    17: "ح",
    18: "خ",
    19: "د",
    20: "ذ",
    21: "ر",
    22: "ز",
    23: "ژ",
    24: "س",
    25: "ش",
    26: "ص",
    27: "ض",
    28: "ط",
    29: "ظ",
    30: "ع",
    31: "غ",
    32: "ف",
    33: "ق",
    34: "ک",
    35: "گ",
    36: "ل",
    37: "م",
    38: "ن",
    39: "و",
    40: "ه",
    41: "ی",
}

LETTER_CLASS_IDS = frozenset(range(10, 42))

# ------------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------------

def normalize(text: str) -> str:
    """Strip whitespace/dashes/ZW chars and unify Arabic/Persian letter variants."""
    if not text:
        return ""
    return (
        text.strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("ي", "ی")    # Arabic ya → Persian ya
        .replace("ك", "ک")    # Arabic kaf → Persian kaf
        .replace("\u200c", "") # zero-width non-joiner
        .replace("\u200b", "") # zero-width space
    )


# ------------------------------------------------------------------
# FORMAT  (display format: "12 ب 345 ایران 67")
# ------------------------------------------------------------------

def format_plate(raw: str) -> str:
    """
    Convert an 8-character raw plate string to the standard Iranian display format.
    Returns empty string if input length is not exactly 8.

    Expected structure:
        positions 0-1 : two digits (region code)
        position  2   : one Persian letter
        positions 3-5 : three digits (serial)
        positions 6-7 : two digits (province code)
    """
    raw = normalize(raw)
    if len(raw) != 8:
        return ""
    return f"{raw[0:2]} {raw[2]} {raw[3:6]} ایران {raw[6:8]}"


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------

# DD space Letter space DDD space ایران space DD
PLATE_REGEX = re.compile(
    r"^\d{2}\s[آ-ی]\s\d{3}\sایران\s\d{2}$"
)


def is_valid_plate(plate: str) -> bool:
    """Return True if the formatted plate string matches the Iranian standard."""
    return bool(PLATE_REGEX.match(plate))


# ------------------------------------------------------------------
# RECONSTRUCTION  (primary pipeline entry point)
# ------------------------------------------------------------------

def reconstruct(detections: List[Dict[str, Any]]) -> str:
    """
    Build a raw plate string from YOLOOCR output.

    Args:
        detections: list of dicts, each with:
            - "class_id": int   — OCR class index (0-indexed, no plate class)
            - "bbox":     list  — [x1, y1, x2, y2] in pixel coords
            - "confidence": float (optional, already filtered upstream)

    Returns:
        Raw plate string, e.g. "12ب34567"
    """
    if not detections:
        return ""
    sorted_dets = sorted(detections, key=lambda d: d["bbox"][0])
    return "".join(CLASS_MAP.get(d["class_id"], "") for d in sorted_dets)


# ------------------------------------------------------------------
# CONVENIENCE HELPERS
# ------------------------------------------------------------------

def extract_plate(detections: List[Dict[str, Any]]) -> str:
    """Reconstruct and format in one call."""
    return format_plate(reconstruct(detections))


def is_valid_iranian_plate(detections: List[Dict[str, Any]]) -> bool:
    """Reconstruct, format, and validate in one call."""
    return is_valid_plate(format_plate(reconstruct(detections)))

from __future__ import annotations
from typing import List, Dict, Any
import re

# ---------------------------
# CLASS MAP (YOLO output)
# ---------------------------
CLASS_MAP = {
    0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
    10:"Ø§Ù„Ù",11:"Ø¨",12:"Ù¾",13:"Øª",
    14:"Ø«",15:"Ø¬",16:"Ú†",17:"Ø­",18:"Ø®",
    19:"Ø¯",20:"Ø°",21:"Ø±",22:"Ø²",23:"Ú˜",
    24:"Ø³",25:"Ø´",26:"Øµ",27:"Ø¶",28:"Ø·",
    29:"Ø¸",30:"Ø¹",31:"Øº",32:"Ù",33:"Ù‚",
    34:"Ú©",35:"Ú¯",36:"Ù„",37:"Ù…",38:"Ù†",
    39:"Ùˆ",40:"Ù‡",41:"ÛŒ",
}

LETTER_CLASS_IDS = frozenset(range(10, 42))

# ---------------------------
# NORMALIZATION (SAFE ONLY)
# ---------------------------
def normalize(text: str) -> str:
    if not text:
        return ""

    return (
        text.strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("ÙŠ", "ÛŒ")
        .replace("Ùƒ", "Ú©")
        .replace("\u200c", "")
    )

# ---------------------------
# FORMAT OUTPUT (DISPLAY FORMAT)
# ---------------------------
def format_plate(raw: str) -> str:
    raw = normalize(raw)

    if len(raw) != 8:
        return ""

    return f"{raw[0:2]} {raw[2]} {raw[3:6]} Ø§ÛŒØ±Ø§Ù† {raw[6:8]}"

# ---------------------------
# REGEX (FINAL VALIDATION)
# ---------------------------
PLATE_REGEX = re.compile(
    r"^\d{2}\s?[Ø¢-ÛŒ]\s?\d{3}\s?Ø§ÛŒØ±Ø§Ù†\s?\d{2}$"
)

def is_valid_plate(plate: str) -> bool:
    return bool(PLATE_REGEX.match(plate))


# ---------------------------
# RECONSTRUCT FROM YOLO
# ---------------------------
def reconstruct(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return ""

    dets = sorted(detections, key=lambda d: d["bbox"][0])

    return "".join(
        CLASS_MAP.get(d.get("class_id"), "")
        for d in dets
    )


# ---------------------------
# FINAL PIPELINE ENTRY
# ---------------------------
def is_valid_iranian_plate(detections: List[Dict[str, Any]]) -> bool:
    raw = reconstruct(detections)
    formatted = format_plate(raw)
    return is_valid_plate(formatted)


def extract_plate(detections: List[Dict[str, Any]]) -> str:
    raw = reconstruct(detections)
    return format_plate(raw)

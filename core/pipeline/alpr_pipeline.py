# -*- coding: utf-8 -*-
"""
alpr_pipeline.py — main ALPR orchestrator.

Fixes applied vs original:
  1. Constructor parameter renamed from 'reconstruct' to 'reconstruct_fn'
     to avoid shadowing the module-level import.
  2. Stored as self._reconstruct_fn and called explicitly in run().
  3. reconstruct import aliased to _reconstruct_default to be safe.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any, Optional

import numpy as np

from core.ocr.reconstruct import (
    reconstruct as _reconstruct_default,
    format_plate,
    is_valid_plate,
)

logger = logging.getLogger(__name__)


class ALPRPipeline:
    """
    Two-stage ALPR pipeline:
      1. Plate detection  → bounding box
      2. Plate cropping   → normalised image region
      3. OCR              → character detections
      4. Reconstruction   → raw plate string
      5. Formatting       → "DD L DDD ایران DD"
      6. Validation       → regex check
    """

    def __init__(
        self,
        detector,
        ocr,
        cropper: Callable[[np.ndarray, list], Optional[np.ndarray]],
        reconstruct_fn: Callable[[list], str] = None,
    ) -> None:
        self.detector        = detector
        self.ocr             = ocr
        self.cropper         = cropper
        # Use injected function if provided, otherwise fall back to default
        self._reconstruct_fn = reconstruct_fn if reconstruct_fn is not None \
                               else _reconstruct_default

    # ------------------------------------------------------------------

    def run(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run the full pipeline on a single frame.

        Args:
            frame: BGR image as np.ndarray (OpenCV format).

        Returns:
            {
                "plate_raw":  str | None,
                "plate":      str | None,
                "valid":      bool,
                "confidence": float,
                "ocr_chars":  int,
            }
        """
        # ── 1. Plate detection ──────────────────────────────────────────────
        detections = self.detector.detect(frame)

        if not detections:
            logger.debug("No plate detected in frame.")
            return {
                "plate_raw":  None,
                "plate":      None,
                "valid":      False,
                "confidence": 0.0,
                "ocr_chars":  0,
            }

        best = max(detections, key=lambda d: d["confidence"])
        logger.debug("Best plate detection: conf=%.3f bbox=%s",
                     best["confidence"], best["bbox"])

        # ── 2. Crop ─────────────────────────────────────────────────────────
        plate_img = self.cropper(frame, best["bbox"])

        if plate_img is None:
            logger.warning("Cropper returned None for bbox %s", best["bbox"])
            return {
                "plate_raw":  None,
                "plate":      None,
                "valid":      False,
                "confidence": round(best["confidence"], 3),
                "ocr_chars":  0,
            }

        # ── 3. OCR ──────────────────────────────────────────────────────────
        char_detections = self.ocr.recognize(plate_img)
        logger.debug("OCR returned %d character detections.", len(char_detections))

        # ── 4. Reconstruction ────────────────────────────────────────────────
        raw_plate = self._reconstruct_fn(char_detections)

        # ── 5. Format + 6. Validate ─────────────────────────────────────────
        formatted_plate = format_plate(raw_plate)
        valid           = is_valid_plate(formatted_plate)

        logger.info("Plate: raw=%r  formatted=%r  valid=%s", raw_plate, formatted_plate, valid)

        return {
            "plate_raw":  raw_plate,
            "plate":      formatted_plate,
            "valid":      valid,
            "confidence": round(best["confidence"], 3),
            "ocr_chars":  len(char_detections),
        }

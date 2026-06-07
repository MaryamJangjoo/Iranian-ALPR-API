# -*- coding: utf-8 -*-
"""
factory.py — builds an ALPRPipeline from the config registry.

Fixes applied vs original:
  1. Import path corrected:  core.config.config_registry  (was core.config_registry)
  2. Cropper import corrected: core.utils.cropper  (was core.cropper)
  3. reconstruct imported for injection into the pipeline.
"""

from core.pipeline.alpr_pipeline import ALPRPipeline
from core.config.config_registry import get_detector_class, get_ocr_class
from core.utils.cropper import crop_plate
from core.ocr.reconstruct import reconstruct


def get_pipeline() -> ALPRPipeline:
    """
    Instantiate and return a fully wired ALPRPipeline.

    Detector and OCR backends are selected via environment variables:
        DETECTOR_TYPE = "yolo" | "vendor"   (default: "yolo")
        OCR_TYPE      = "yolo" | "vendor"   (default: "yolo")

    Model paths and confidence thresholds are read from core/config/config.py,
    which itself respects environment variable overrides.
    """
    detector = get_detector_class()()
    ocr      = get_ocr_class()()

    return ALPRPipeline(
        detector=detector,
        ocr=ocr,
        cropper=crop_plate,
        reconstruct_fn=reconstruct,
    )

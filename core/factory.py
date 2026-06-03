"""
factory.py — builds ALPRPipeline from config_registry
"""

from core.pipeline.alpr_pipeline import ALPRPipeline
from core.config_registry import get_detector_class, get_ocr_class
from core.cropper import crop_plate
from core.ocr.reconstruct import reconstruct


def get_pipeline() -> ALPRPipeline:
    detector = get_detector_class()()
    ocr = get_ocr_class()()

    return ALPRPipeline(
        detector=detector,
        ocr=ocr,
        cropper=crop_plate,
        reconstruct=reconstruct,
    )


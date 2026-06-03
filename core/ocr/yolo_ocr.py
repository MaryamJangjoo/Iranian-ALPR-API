"""
YOLOv8-based character OCR â€” Stage 2 of ALPR pipeline
Clean production version
"""

from ultralytics import YOLO
from core.interfaces.ocr import BaseOCR
from core.config import OCR_MODEL_PATH, OCR_CONF, DEVICE


class YOLOOCR(BaseOCR):

    def __init__(self):
        if not OCR_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"OCR model not found: {OCR_MODEL_PATH}"
            )

        self.model = YOLO(str(OCR_MODEL_PATH))

    def recognize(self, image):
        """
        Returns:
            List of character detections:
            [
                {
                    "class_id": int,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        """

        results = self.model.predict(
            source=image,
            conf=0.1,        # OCR threshold (you can tune)
            iou=0.3,
            device=DEVICE,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        detections = []

        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        detections = [d for d in detections if d["confidence"] > 0.25]
        detections.sort(key=lambda x: x["bbox"][0]) 
        return detections


def get_ocr() -> YOLOOCR:
    return YOLOOCR()

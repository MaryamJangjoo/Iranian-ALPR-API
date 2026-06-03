"""
YOLOv8-based plate detector — Stage 1 of the ALPR pipeline.
"""

from ultralytics import YOLO
from core.interfaces.detector import BaseDetector
from core.config import PLATE_MODEL_PATH, PLATE_CONF, DEVICE


class YOLOPlateDetector(BaseDetector):

    def __init__(self):
        if not PLATE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Plate model not found: {PLATE_MODEL_PATH}"
            )
        self.model = YOLO(str(PLATE_MODEL_PATH))

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=PLATE_CONF,
            device=DEVICE,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:

            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # ?? FIX 1: padding (VERY IMPORTANT)
            pad = 8
            h, w = frame.shape[:2]

            x1 = max(0, int(x1 - pad))
            y1 = max(0, int(y1 - pad))
            x2 = min(w, int(x2 + pad))
            y2 = min(h, int(y2 + pad))

            conf = float(box.conf[0])

            # ?? FIX 2: confidence filtering
            if conf < 0.3:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            })

        # ?? FIX 3: if multiple plates ? pick best
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return detections

def get_detector() -> YOLOPlateDetector:
    return YOLOPlateDetector()

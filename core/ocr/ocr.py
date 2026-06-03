from core.models import get_ocr_model
from core.config import OCR_CONF

ocr_model = get_ocr_model()


def ocr_plate(plate_crop):
    """
    Run OCR model on cropped plate image

    Returns:
        list of detections
    """

    if plate_crop is None:
        return []

    results = ocr_model.predict(
        source=plate_crop,
        conf=OCR_CONF,
        verbose=False
    )

    if len(results) == 0:
        return []

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        cls_id = int(box.cls[0])

        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        })

    return detections

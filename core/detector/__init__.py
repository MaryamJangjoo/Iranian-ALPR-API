from .plate_detector import YOLOPlateDetector as PlateDetector

_detector = None


def get_detector():
    """
    Lazy-load YOLO detector (prevents crash on import)
    """
    global _detector

    if _detector is None:
        _detector = PlateDetector()

    return _detector


def detect_plate(frame):
    """
    Wrapper for YOLO Plate Detection
    """
    detector = get_detector()
    return detector.detect(frame)

from .plate_detector import YOLOPlateDetector as PlateDetector

_detector = PlateDetector()

def detect_plate(frame):
    """
    Wrapper for YOLO Plate Detection
    """
    return _detector.detect(frame)

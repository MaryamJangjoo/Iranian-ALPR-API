import cv2
from ALPR_system.inference.pipeline import ALPRPipeline

# مسیر مدل‌ها
plate_model = "models/plate_detector.pt"
ocr_model = "models/ocr_detector.pt"

engine = ALPRPipeline(plate_model, ocr_model)

img = cv2.imread("test.jpg")

result = engine.predict(img)

print("RESULT:", result)

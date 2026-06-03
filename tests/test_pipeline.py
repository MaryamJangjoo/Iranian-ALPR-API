import cv2
from core.factory import get_pipeline

pipeline = get_pipeline()

img_path = "/home/developer/LPR/IR-LPR-main/Car Image/car_img-test/day_00019.jpg"
img = cv2.imread(img_path)

print("Image loaded:", img is not None)

result = pipeline.run(img)

print("RESULT:", result)

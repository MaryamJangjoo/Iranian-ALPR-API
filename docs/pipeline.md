# ALPR Pipeline

## Steps

1. Input image received
2. YOLO detects license plate
3. Plate is cropped
4. OCR extracts characters
5. Reconstruction formats plate
6. Validation is applied
7. Result returned via API

---

## Flow Diagram

Image → Detection → OCR → Format → Output

from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2
import time

from core.factory import get_pipeline

router = APIRouter()

# ---------------------------
# SINGLETON PIPELINE (IMPORTANT)
# ---------------------------
pipeline = get_pipeline()


@router.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    start = time.time()

    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="invalid_image"
        )

    # ---------------------------
    # RUN PIPELINE
    # ---------------------------
    result = pipeline.run(frame)

    # ---------------------------
    # RESPONSE WRAP (SDK STYLE)
    # ---------------------------
    return {
        "success": True,
        "plate": result.get("plate"),
        "plate_raw": result.get("plate_raw"),
        "valid": result.get("valid"),
        "confidence": result.get("confidence", 0.0),
        "api_latency_ms": int((time.time() - start) * 1000)
    }
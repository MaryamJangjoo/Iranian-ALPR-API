# ALPR API Documentation

## Overview
This API performs Automatic License Plate Recognition (ALPR) for Iranian vehicles using:
- YOLOv8-based plate detector
- Dual OCR engine (YOLO OCR + vendor OCR fallback)
- Persian license plate reconstruction

**Base URL:** `http://localhost:8000`

---

## Endpoint

### POST /recognize

Upload an image and get the detected Iranian license plate.

---

## Request

- **Content-Type:** `multipart/form-data`
- **Method:** `POST`

### Form Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | image file | ✅ Yes | Image containing a vehicle with visible Iranian license plate |

### Supported Image Formats

- JPEG/JPG
- PNG
- BMP

**Max file size:** 10 MB

---

## Response

### Success Response (200 OK)

```json
{
  "success": true,
  "plate": "95 م 881 ایران 99",
  "plate_raw": "95م88199",
  "valid": true,
  "confidence": 0.85,
  "api_latency_ms": 293
}

# 🚗 Iranian ALPR API

> **Automatic License Plate Recognition** for Iranian license plates — built with YOLOv8, FastAPI, and Clean Architecture.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://ultralytics.com/)

---

## 📌 Overview

**Iranian-ALPR-API** is a production-ready REST API that detects and reads Iranian license plates from images. It uses a two-stage YOLOv8 pipeline: first to locate the plate, then to recognize each character — outputting a validated, formatted plate string in one API call.

```
Input: car image  →  Output: { "plate": "12 ب 345 ایران 67", "valid": true }
```

---

## ✨ Features

- 🔍 **Two-stage YOLO pipeline** — plate detection + character-level OCR
- 🇮🇷 **Persian plate reconstruction** — correct right-to-left character ordering and formatting
- ✅ **Plate validation** — regex-based format check on the final output
- 🔄 **Vendor OCR fallback** — pluggable secondary OCR engine via Strategy pattern
- ⚙️ **Clean Architecture** — interfaces, dependency injection, factory pattern
- 🐳 **Docker ready** — single container deployment on port 8000
- 🔧 **Fully configurable** — all thresholds and model paths via environment variables

---

## 🏗️ Architecture

```
Client
  │ HTTP POST /recognize
  ▼
FastAPI Layer          ← request validation, response serialization
  │
  ▼
ALPR Pipeline          ← orchestrator (detect → crop → OCR → reconstruct → validate)
  │
  ├── YOLOv8 Detector  ← locates plate bounding box
  ├── Cropper          ← extracts and normalizes plate region
  ├── YOLO OCR         ← character detection (with vendor fallback)
  └── Reconstructor    ← builds + validates Persian plate string
```

See [docs/architecture.md](docs/architecture.md) for the full system design.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- YOLOv8 model weights (see [Models](#-models) below)

### 1. Clone & Install

```bash
git clone https://github.com/MaryamJangjoo/Iranian-ALPR-API.git
cd Iranian-ALPR-API

pip install -r requirements/base.txt
pip install ultralytics torch torchvision
```

### 2. Add Model Weights

Place your trained model files in `core/models/`:

```
core/
└── models/
    ├── plate_detector.pt   ← YOLOv8 plate detection model
    └── ocr_detector.pt     ← YOLOv8 character OCR model
```

### 3. Run the API

```bash
python run.py
```

API is now live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
docker build -t iranian-alpr-api .
docker run -p 8000:8000 iranian-alpr-api
```

---

## 📡 API Reference

### `POST /recognize`

Detect and read a license plate from an uploaded image.

**Request**

```bash
curl -X POST http://localhost:8000/recognize \
  -F "file=@car.jpg"
```

**Response**

```json
{
  "success": true,
  "plate": "12 ب 345 ایران 67",
  "plate_raw": "12ب34567",
  "valid": true,
  "confidence": 0.924,
  "api_latency_ms": 87
}
```

| Field | Type | Description |
|---|---|---|
| `plate` | `string` | Formatted Persian plate string |
| `plate_raw` | `string` | Raw reconstructed characters |
| `valid` | `bool` | Whether plate matches expected format |
| `confidence` | `float` | Detection confidence score (0–1) |
| `api_latency_ms` | `int` | End-to-end API latency in milliseconds |

### `GET /health`

```json
{ "status": "ok", "service": "iranian-alpr-api" }
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `PLATE_MODEL_PATH` | `core/models/plate_detector.pt` | Path to plate detection model |
| `OCR_MODEL_PATH` | `core/models/ocr_detector.pt` | Path to OCR model |
| `PLATE_CONF` | `0.4` | Minimum confidence for plate detection |
| `OCR_CONF` | `0.25` | Minimum confidence for character recognition |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`) |
| `DETECTOR_TYPE` | `yolo` | Detector backend (`yolo` or `vendor`) |
| `OCR_TYPE` | `yolo` | OCR backend (`yolo` or `vendor`) |

---
## 🧪 Testing

Run all automated tests:

```bash
pytest tests/
```

Run the pipeline integration test:

```bash
python tests/test_pipeline.py
```

Run the end-to-end ALPR test:

```bash
python tests/test_alpr.py
```

### Test Files

| File | Purpose |
|--------|--------|
| `test_pipeline.py` | Tests the full ALPR pipeline using the factory-created pipeline instance |
| `test_alpr.py` | End-to-end ALPR inference test using trained YOLO models |


---

## 📁 Project Structure

```
Iranian-ALPR-API/
├── api/
│   ├── main.py            # FastAPI app entry point
│   ├── routes.py          # API route definitions
│   └── recognize.py       # Response serialization
├── core/
│   ├── config/
│   │   ├── config.py          # Central configuration
│   │   └── config_registry.py # Model backend selection
│   ├── detector/
│   │   └── plate_detector.py  # YOLOv8 plate detector
│   ├── interfaces/
│   │   ├── detector.py        # BaseDetector ABC
│   │   └── ocr.py             # BaseOCR ABC
│   ├── ocr/
│   │   ├── yolo_ocr.py        # YOLO character OCR
│   │   ├── vendor_ocr.py      # Fallback OCR engine
│   │   └── reconstruct.py     # Persian plate formatting
│   ├── pipeline/
│   │   └── alpr_pipeline.py   # Main orchestrator
│   ├── utils/
│   │   └── cropper.py         # Image cropping utilities
│   └── factory.py             # Pipeline factory
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── pipeline.md
├── tests/
│   ├── test_pipeline.py
│   └── test_alpr.py
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── run.py
└── README.md
```

---

## 🧩 Models

This repository does **not** include model weights (they are excluded via `.gitignore`).

To train your own models:
- **Plate Detector** — train YOLOv8 on an Iranian license plate dataset (e.g. [OpenALPR datasets](https://github.com/openalpr/benchmarks))
- **OCR Model** — train YOLOv8 on cropped plate images with per-character bounding box annotations (42 classes: digits 0–9 + 32 Persian letters)

---

## 🛣️ Roadmap

- [ ] Batch image processing endpoint
- [ ] Model versioning support
- [ ] Redis response caching
- [ ] Prometheus + Grafana monitoring
- [ ] Kubernetes deployment manifests
- [ ] Pre-trained model release

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---



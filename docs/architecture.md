# ALPR System Architecture

## Overview

This document describes the architecture of the Iranian ALPR system.
The system follows **Clean Architecture** principles with separation of concerns, dependency inversion, and interface-based design.

---

## High-Level Architecture

```text
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   FastAPI   │  ← API Layer (/api)
│    Layer    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Pipeline   │  ← Orchestrator
│ Orchestrator│    (alpr_pipeline.py)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│         Core Layer              │
│  ┌───────────┐  ┌─────────────┐ │
│  │ Detector  │  │    OCR      │ │
│  │ (YOLOv8)  │  │ (YOLO/Vendor)│ │
│  └───────────┘  └─────────────┘ │
│  ┌───────────┐  ┌─────────────┐ │
│  │  Cropper  │  │Reconstruct  │ │
│  └───────────┘  └─────────────┘ │
└─────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   JSON      │
│  Response   │
└─────────────┘
```

---

## Layer Descriptions

### API Layer (`/api`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app initialization, CORS, middleware |
| `routes.py` | API route definitions |
| `recognize.py` | Request validation, response serialization |

### Pipeline Layer (`/core/pipeline`)

| File | Responsibility |
|---|---|
| `alpr_pipeline.py` | Main orchestrator: load → detect → crop → OCR → reconstruct → validate |


#### Interfaces

```python
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List:
        pass

class OCR(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        pass
```

#### Detector Module (`/core/detector/`)

| File | Responsibility |
|---|---|
| `plate_detector.py` | YOLOv8 license plate detection |

#### OCR Module (`/core/ocr/`)

| File | Responsibility |
|---|---|
| `yolo_ocr.py` | YOLO-based character recognition |
| `vendor_ocr.py` | Fallback OCR engine |
| `reconstruct.py` | Persian license plate formatting |

#### Utils (`/core/utils/`)

| File | Responsibility |
|---|---|
| `cropper.py` | Image cropping utilities |

---

## Data Flow

```text
1. Image upload (HTTP)
2. API validation
3. YOLO plate detection
4. Crop plate region
5. OCR recognition (YOLO → Vendor fallback)
6. Persian reconstruction
7. Format validation
8. JSON response
```

---

## Design Patterns

| Pattern | Location | Purpose |
|---|---|---|
| Factory | `core/factory.py` | Dynamic model loading |
| Strategy | Interface implementations | OCR switching |
| Registry | `core/config/config_registry.py` | Configuration management |
| Dependency Injection | Interfaces | Loose coupling |

---

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| Detection / OCR | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Image Processing | OpenCV, NumPy |
| Containerization | Docker |

---

## Error Handling

```text
Request
    ↓
Pipeline → Custom Exceptions (NoPlateDetected, OCRFailure)
    ↓
Detector/OCR → Graceful degradation with fallbacks
```

---

## Configuration Management

Supports multi-environment config via:

- Environment variables (highest priority)
- `.env` file
- Default values

| Environment | Requirements File |
|---|---|
| Development | `requirements/dev.txt` |
| Production | `requirements/prod.txt` |

---

## Performance

| Component | Optimization |
|---|---|
| Models | Lazy loading, GPU acceleration |
| API | Async endpoints |
| Processing | Efficient NumPy operations |

---

## Deployment

```text
┌─────────────────────────┐
│    Docker Container     │
│  ┌───────────────────┐  │
│  │ FastAPI (Uvicorn) │  │
│  │     Port 8000     │  │
│  ├───────────────────┤  │
│  │  YOLO Models      │  │
│  │  (in memory)      │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

---

## Future Improvements

- [ ] Model versioning
- [ ] Batch processing
- [ ] Redis caching
- [ ] Prometheus monitoring
- [ ] Kubernetes deployment

---

## Related Docs

- [API Documentation](api.md)
- [Pipeline Documentation](pipeline.md)

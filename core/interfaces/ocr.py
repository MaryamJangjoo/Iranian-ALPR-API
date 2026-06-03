"""
Abstract base class for OCR engines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseOCR(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Returns:
            [{"class_id": int, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """

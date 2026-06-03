"""
Abstract base class for plate detectors.
Import this in detector implementations — never the other way round.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float
                }
            ]
        """
        pass

"""3-class health classification on top of RUL predictions.

Classes:
- HEALTHY: RUL > 50
- WARNING: 20 <= RUL <= 50
- CRITICAL: RUL < 20
"""

from __future__ import annotations

import numpy as np

# Thresholds (in cycles)
HEALTHY_THRESHOLD = 50
CRITICAL_THRESHOLD = 20

CLASSES = ["HEALTHY", "WARNING", "CRITICAL"]


def rul_to_class(rul: float) -> str:
    """Convert a single RUL value to a health class."""
    if rul > HEALTHY_THRESHOLD:
        return "HEALTHY"
    if rul >= CRITICAL_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


def ruls_to_classes(ruls) -> np.ndarray:
    """Vectorized RUL → class conversion."""
    ruls = np.asarray(ruls, dtype=float)
    classes = np.empty(len(ruls), dtype=object)
    classes[ruls > HEALTHY_THRESHOLD] = "HEALTHY"
    classes[(ruls >= CRITICAL_THRESHOLD) & (ruls <= HEALTHY_THRESHOLD)] = "WARNING"
    classes[ruls < CRITICAL_THRESHOLD] = "CRITICAL"
    return classes


def class_to_risk(class_name: str) -> str:
    """Map class to risk level string."""
    return {
        "HEALTHY": "low",
        "WARNING": "medium",
        "CRITICAL": "high",
    }.get(class_name, "unknown")
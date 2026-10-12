"""Tests for RUL → class conversion."""

import numpy as np

from app.pipeline.classification import (
    CLASSES,
    CRITICAL_THRESHOLD,
    HEALTHY_THRESHOLD,
    class_to_risk,
    rul_to_class,
    ruls_to_classes,
)


def test_healthy_classification():
    assert rul_to_class(100) == "HEALTHY"
    assert rul_to_class(51) == "HEALTHY"


def test_warning_classification():
    assert rul_to_class(50) == "WARNING"
    assert rul_to_class(30) == "WARNING"
    assert rul_to_class(20) == "WARNING"


def test_critical_classification():
    assert rul_to_class(19) == "CRITICAL"
    assert rul_to_class(0) == "CRITICAL"


def test_thresholds_match():
    assert HEALTHY_THRESHOLD == 50
    assert CRITICAL_THRESHOLD == 20


def test_vectorized_conversion():
    ruls = np.array([100, 50, 30, 20, 10, 0])
    expected = np.array(["HEALTHY", "WARNING", "WARNING", "WARNING", "CRITICAL", "CRITICAL"])
    result = ruls_to_classes(ruls)
    assert list(result) == list(expected)


def test_classes_list():
    assert CLASSES == ["HEALTHY", "WARNING", "CRITICAL"]


def test_class_to_risk():
    assert class_to_risk("HEALTHY") == "low"
    assert class_to_risk("WARNING") == "medium"
    assert class_to_risk("CRITICAL") == "high"
    assert class_to_risk("UNKNOWN") == "unknown"
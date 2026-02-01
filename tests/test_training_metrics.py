from __future__ import annotations

import numpy as np

from src.train import training


def test_sanitize_metric_value_replaces_non_finite() -> None:
    assert training._sanitize_metric_value("metric", np.nan) == 0.0
    assert training._sanitize_metric_value("metric", np.inf) == 0.0
    assert training._sanitize_metric_value("metric", -np.inf) == 0.0


def test_compute_cafa_metrics_returns_finite_values() -> None:
    probabilities = np.array([[np.nan, 0.2], [0.3, np.nan]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = training.compute_cafa_metrics(
        probabilities=probabilities,
        targets=targets,
        thresholds=[0.5],
        ia_weights=None,
    )
    assert metrics
    assert all(np.isfinite(value) for value in metrics.values())

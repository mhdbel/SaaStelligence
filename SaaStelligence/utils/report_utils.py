"""Utilities for presenting reporting data."""

from typing import Any, Dict


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Round float metrics for presentation while keeping other types intact."""

    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = round(value, 4 if key in {"CTR", "CVR"} else 2)
        else:
            formatted[key] = value
    return formatted
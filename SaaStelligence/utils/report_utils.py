"""
Utilities for formatting and presenting reporting data.

This module provides functions for:
- Formatting metrics for display (percentages, currency, etc.)
- Generating report summaries
- Exporting reports to various formats
- Date/time formatting for reports

Example:
    >>> from utils.report_utils import format_metrics, format_as_table
    >>> 
    >>> metrics = {"CTR": 0.0234, "CVR": 0.0512, "CPA": 45.678, "Leads": 150}
    >>> formatted = format_metrics(metrics)
    >>> print(formatted)
    {'CTR': '2.34%', 'CVR': '5.12%', 'CPA': '$45.68', 'Leads': 150}
    >>> 
    >>> print(format_as_table(formatted))
    ┌─────────┬─────────┐
    │ Metric  │ Value   │
    ├─────────┼─────────┤
    │ CTR     │ 2.34%   │
    │ CVR     │ 5.12%   │
    │ CPA     │ $45.68  │
    │ Leads   │ 150     │
    └─────────┴─────────┘
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

# ============== LOGGING ==============
logger = logging.getLogger(__name__)


# ============== CONSTANTS ==============
class MetricType(Enum):
    """Types of metrics for formatting purposes."""
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DURATION = "duration"


# Default metric type mappings
DEFAULT_METRIC_TYPES: Dict[str, MetricType] = {
    "CTR": MetricType.PERCENTAGE,
    "CVR": MetricType.PERCENTAGE,
    "conversion_rate": MetricType.PERCENTAGE,
    "click_through_rate": MetricType.PERCENTAGE,
    "bounce_rate": MetricType.PERCENTAGE,
    "CPA": MetricType.CURRENCY,
    "cost": MetricType.CURRENCY,
    "revenue": MetricType.CURRENCY,
    "spend": MetricType.CURRENCY,
    "budget": MetricType.CURRENCY,
    "bid": MetricType.CURRENCY,
    "Leads": MetricType.INTEGER,
    "leads": MetricType.INTEGER,
    "clicks": MetricType.INTEGER,
    "impressions": MetricType.INTEGER,
    "conversions": MetricType.INTEGER,
    "sessions": MetricType.INTEGER,
    "users": MetricType.INTEGER,
    "processing_time_ms": MetricType.FLOAT,
    "latency": MetricType.FLOAT,
    "confidence": MetricType.PERCENTAGE,
    "intent_confidence": MetricType.PERCENTAGE,
    "predicted_cvr": MetricType.PERCENTAGE,
}


# ============== CONFIGURATION ==============
@dataclass
class FormatConfig:
    """Configuration for metric formatting."""
    
    # Percentage formatting
    percentage_decimals: int = 2
    percentage_symbol: str = "%"
    percentage_multiply: bool = True  # Multiply by 100
    
    # Currency formatting
    currency_symbol: str = "$"
    currency_decimals: int = 2
    currency_position: str = "before"  # "before" or "after"
    thousands_separator: str = ","
    
    # Number formatting
    float_decimals: int = 4
    integer_separator: str = ","
    
    # Display options
    include_metric_type: bool = False
    null_display: str = "N/A"
    zero_display: Optional[str] = None  # None means show "0"
    
    # For raw values (non-display)
    raw_mode: bool = False


# Default configuration
DEFAULT_CONFIG = FormatConfig()


# ============== FORMATTING FUNCTIONS ==============

def format_percentage(
    value: float,
    decimals: int = 2,
    symbol: str = "%",
    multiply: bool = True,
) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: The value to format (0.05 = 5% if multiply=True).
        decimals: Number of decimal places.
        symbol: Percentage symbol to append.
        multiply: Whether to multiply by 100.
        
    Returns:
        Formatted percentage string.
        
    Example:
        >>> format_percentage(0.0523)
        '5.23%'
        >>> format_percentage(5.23, multiply=False)
        '5.23%'
    """
    if value is None:
        return DEFAULT_CONFIG.null_display
    
    display_value = value * 100 if multiply else value
    return f"{display_value:.{decimals}f}{symbol}"


def format_currency(
    value: float,
    symbol: str = "$",
    decimals: int = 2,
    position: str = "before",
    thousands_sep: str = ",",
) -> str:
    """
    Format a value as currency.
    
    Args:
        value: The monetary value.
        symbol: Currency symbol.
        decimals: Decimal places.
        position: Symbol position ("before" or "after").
        thousands_sep: Thousands separator character.
        
    Returns:
        Formatted currency string.
        
    Example:
        >>> format_currency(1234.567)
        '$1,234.57'
        >>> format_currency(1234.56, symbol="€", position="after")
        '1,234.56€'
    """
    if value is None:
        return DEFAULT_CONFIG.null_display
    
    # Format with decimals
    formatted = f"{abs(value):,.{decimals}f}"
    
    # Replace comma with custom separator if different
    if thousands_sep != ",":
        formatted = formatted.replace(",", thousands_sep)
    
    # Handle negative values
    sign = "-" if value < 0 else ""
    
    if position == "before":
        return f"{sign}{symbol}{formatted}"
    else:
        return f"{sign}{formatted}{symbol}"


def format_integer(
    value: int,
    thousands_sep: str = ",",
) -> str:
    """
    Format an integer with thousands separator.
    
    Args:
        value: The integer value.
        thousands_sep: Separator character.
        
    Returns:
        Formatted integer string.
        
    Example:
        >>> format_integer(1234567)
        '1,234,567'
    """
    if value is None:
        return DEFAULT_CONFIG.null_display
    
    formatted = f"{int(value):,}"
    
    if thousands_sep != ",":
        formatted = formatted.replace(",", thousands_sep)
    
    return formatted


def format_float(
    value: float,
    decimals: int = 4,
) -> str:
    """
    Format a float with specified precision.
    
    Args:
        value: The float value.
        decimals: Number of decimal places.
        
    Returns:
        Formatted float string.
    """
    if value is None:
        return DEFAULT_CONFIG.null_display
    
    return f"{value:.{decimals}f}"


def format_duration(
    seconds: float,
    precision: str = "auto",
) -> str:
    """
    Format a duration in human-readable form.
    
    Args:
        seconds: Duration in seconds.
        precision: "auto", "seconds", "milliseconds", or "minutes".
        
    Returns:
        Formatted duration string.
        
    Example:
        >>> format_duration(125.5)
        '2m 5.5s'
        >>> format_duration(0.0234)
        '23.4ms'
    """
    if seconds is None:
        return DEFAULT_CONFIG.null_display
    
    if precision == "milliseconds" or (precision == "auto" and seconds < 1):
        return f"{seconds * 1000:.1f}ms"
    elif precision == "minutes" or (precision == "auto" and seconds >= 60):
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.1f}s"
        return f"{minutes}m"
    else:
        return f"{seconds:.2f}s"


def detect_metric_type(key: str, value: Any) -> MetricType:
    """
    Detect the appropriate metric type for a key/value pair.
    
    Args:
        key: The metric name.
        value: The metric value.
        
    Returns:
        Detected MetricType.
    """
    # Check explicit mappings first
    key_lower = key.lower()
    
    if key in DEFAULT_METRIC_TYPES:
        return DEFAULT_METRIC_TYPES[key]
    
    # Check by key patterns
    if any(pattern in key_lower for pattern in ["rate", "ratio", "percent", "pct"]):
        return MetricType.PERCENTAGE
    
    if any(pattern in key_lower for pattern in ["cost", "price", "revenue", "spend", "budget", "cpa", "cpc"]):
        return MetricType.CURRENCY
    
    if any(pattern in key_lower for pattern in ["count", "total", "num", "leads", "clicks", "impressions"]):
        return MetricType.INTEGER
    
    if any(pattern in key_lower for pattern in ["time", "duration", "latency", "_ms", "_seconds"]):
        return MetricType.DURATION
    
    # Fallback based on value type
    if isinstance(value, bool):
        return MetricType.STRING
    elif isinstance(value, int):
        return MetricType.INTEGER
    elif isinstance(value, float):
        # Small floats that look like rates
        if 0 <= value <= 1:
            return MetricType.PERCENTAGE
        return MetricType.FLOAT
    else:
        return MetricType.STRING


def format_value(
    key: str,
    value: Any,
    config: Optional[FormatConfig] = None,
    metric_type: Optional[MetricType] = None,
) -> Any:
    """
    Format a single value based on its metric type.
    
    Args:
        key: The metric name.
        value: The value to format.
        config: Formatting configuration.
        metric_type: Override automatic type detection.
        
    Returns:
        Formatted value (string for display, original type for raw mode).
    """
    cfg = config or DEFAULT_CONFIG
    
    # Handle None
    if value is None:
        return cfg.null_display if not cfg.raw_mode else None
    
    # Handle zero display
    if value == 0 and cfg.zero_display is not None:
        return cfg.zero_display if not cfg.raw_mode else 0
    
    # Raw mode - just round floats
    if cfg.raw_mode:
        if isinstance(value, float):
            detected_type = metric_type or detect_metric_type(key, value)
            if detected_type == MetricType.PERCENTAGE:
                return round(value, cfg.percentage_decimals + 2)  # Keep precision for rates
            elif detected_type == MetricType.CURRENCY:
                return round(value, cfg.currency_decimals)
            else:
                return round(value, cfg.float_decimals)
        return value
    
    # Detect type if not provided
    if metric_type is None:
        metric_type = detect_metric_type(key, value)
    
    # Format based on type
    if metric_type == MetricType.PERCENTAGE:
        return format_percentage(
            float(value),
            decimals=cfg.percentage_decimals,
            symbol=cfg.percentage_symbol,
            multiply=cfg.percentage_multiply,
        )
    
    elif metric_type == MetricType.CURRENCY:
        return format_currency(
            float(value),
            symbol=cfg.currency_symbol,
            decimals=cfg.currency_decimals,
            position=cfg.currency_position,
            thousands_sep=cfg.thousands_separator,
        )
    
    elif metric_type == MetricType.INTEGER:
        return format_integer(
            int(value) if isinstance(value, (int, float)) else value,
            thousands_sep=cfg.integer_separator,
        )
    
    elif metric_type == MetricType.FLOAT:
        return format_float(float(value), decimals=cfg.float_decimals)
    
    elif metric_type == MetricType.DURATION:
        return format_duration(float(value))
    
    else:
        return str(value)


def format_metrics(
    metrics: Dict[str, Any],
    config: Optional[FormatConfig] = None,
    metric_types: Optional[Dict[str, MetricType]] = None,
) -> Dict[str, Any]:
    """
    Format a dictionary of metrics for presentation.
    
    This is the main function for formatting report metrics.
    It automatically detects metric types and applies appropriate formatting.
    
    Args:
        metrics: Dictionary of metric names to values.
        config: Optional formatting configuration.
        metric_types: Optional explicit type mappings.
        
    Returns:
        Dictionary with formatted values.
        
    Example:
        >>> metrics = {"CTR": 0.0234, "CVR": 0.0512, "CPA": 45.678, "Leads": 150}
        >>> format_metrics(metrics)
        {'CTR': '2.34%', 'CVR': '5.12%', 'CPA': '$45.68', 'Leads': '150'}
        
        >>> # Raw mode (for API responses)
        >>> format_metrics(metrics, config=FormatConfig(raw_mode=True))
        {'CTR': 0.0234, 'CVR': 0.0512, 'CPA': 45.68, 'Leads': 150}
    """
    cfg = config or DEFAULT_CONFIG
    type_map = metric_types or {}
    
    formatted: Dict[str, Any] = {}
    
    for key, value in metrics.items():
        metric_type = type_map.get(key)
        formatted[key] = format_value(key, value, config=cfg, metric_type=metric_type)
    
    return formatted


def format_metrics_raw(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for API responses (numeric values, appropriate rounding).
    
    This is a convenience wrapper around format_metrics with raw_mode=True.
    
    Args:
        metrics: Dictionary of metrics.
        
    Returns:
        Dictionary with rounded numeric values.
    """
    return format_metrics(metrics, config=FormatConfig(raw_mode=True))


# ============== REPORT GENERATION ==============

def format_as_table(
    data: Dict[str, Any],
    title: Optional[str] = None,
    key_header: str = "Metric",
    value_header: str = "Value",
) -> str:
    """
    Format a dictionary as an ASCII table.
    
    Args:
        data: Dictionary to display.
        title: Optional table title.
        key_header: Header for key column.
        value_header: Header for value column.
        
    Returns:
        ASCII table string.
        
    Example:
        >>> print(format_as_table({"CTR": "2.34%", "CVR": "5.12%"}))
        ┌────────┬────────┐
        │ Metric │ Value  │
        ├────────┼────────┤
        │ CTR    │ 2.34%  │
        │ CVR    │ 5.12%  │
        └────────┴────────┘
    """
    if not data:
        return "(empty)"
    
    # Calculate column widths
    key_width = max(len(str(k)) for k in data.keys())
    key_width = max(key_width, len(key_header))
    
    value_width = max(len(str(v)) for v in data.values())
    value_width = max(value_width, len(value_header))
    
    # Build table
    lines = []
    
    # Title
    if title:
        total_width = key_width + value_width + 5
        lines.append(f"{'─' * total_width}")
        lines.append(f" {title.center(total_width - 2)} ")
        lines.append(f"{'─' * total_width}")
    
    # Top border
    lines.append(f"┌{'─' * (key_width + 2)}┬{'─' * (value_width + 2)}┐")
    
    # Header
    lines.append(f"│ {key_header:<{key_width}} │ {value_header:<{value_width}} │")
    
    # Header separator
    lines.append(f"├{'─' * (key_width + 2)}┼{'─' * (value_width + 2)}┤")
    
    # Data rows
    for key, value in data.items():
        lines.append(f"│ {str(key):<{key_width}} │ {str(value):<{value_width}} │")
    
    # Bottom border
    lines.append(f"└{'─' * (key_width + 2)}┴{'─' * (value_width + 2)}┘")
    
    return "\n".join(lines)


def format_as_markdown_table(
    data: Dict[str, Any],
    key_header: str = "Metric",
    value_header: str = "Value",
) -> str:
    """
    Format a dictionary as a Markdown table.
    
    Args:
        data: Dictionary to display.
        key_header: Header for key column.
        value_header: Header for value column.
        
    Returns:
        Markdown table string.
    """
    if not data:
        return "*No data*"
    
    lines = [
        f"| {key_header} | {value_header} |",
        "|---|---|",
    ]
    
    for key, value in data.items():
        lines.append(f"| {key} | {value} |")
    
    return "\n".join(lines)


def generate_summary_report(
    metrics: Dict[str, Any],
    title: str = "Performance Report",
    include_timestamp: bool = True,
    format_type: str = "text",
) -> str:
    """
    Generate a formatted summary report.
    
    Args:
        metrics: Dictionary of metrics.
        title: Report title.
        include_timestamp: Whether to include generation timestamp.
        format_type: "text", "markdown", or "json".
        
    Returns:
        Formatted report string.
    """
    formatted = format_metrics(metrics)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    if format_type == "json":
        report_data = {
            "title": title,
            "generated_at": timestamp if include_timestamp else None,
            "metrics": formatted,
        }
        return json.dumps(report_data, indent=2)
    
    elif format_type == "markdown":
        lines = [f"# {title}"]
        if include_timestamp:
            lines.append(f"\n*Generated: {timestamp}*\n")
        lines.append(format_as_markdown_table(formatted))
        return "\n".join(lines)
    
    else:  # text
        lines = [
            "=" * 50,
            title.center(50),
            "=" * 50,
        ]
        if include_timestamp:
            lines.append(f"Generated: {timestamp}")
            lines.append("-" * 50)
        lines.append(format_as_table(formatted))
        return "\n".join(lines)


# ============== EXPORT FUNCTIONS ==============

def export_to_csv(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    format_values: bool = True,
) -> str:
    """
    Export a list of dictionaries to CSV format.
    
    Args:
        data: List of data dictionaries.
        columns: Column order (defaults to keys from first row).
        format_values: Whether to apply metric formatting.
        
    Returns:
        CSV string.
    """
    if not data:
        return ""
    
    # Determine columns
    if columns is None:
        columns = list(data[0].keys())
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    
    for row in data:
        if format_values:
            row = format_metrics(row)
        writer.writerow(row)
    
    return output.getvalue()


def export_to_json(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    format_values: bool = True,
    pretty: bool = True,
) -> str:
    """
    Export data to JSON format.
    
    Args:
        data: Data to export.
        format_values: Whether to apply metric formatting.
        pretty: Whether to pretty-print.
        
    Returns:
        JSON string.
    """
    if format_values:
        if isinstance(data, list):
            data = [format_metrics_raw(item) for item in data]
        else:
            data = format_metrics_raw(data)
    
    if pretty:
        return json.dumps(data, indent=2, default=str)
    return json.dumps(data, default=str)


# ============== COMPARISON UTILITIES ==============

def calculate_change(
    current: float,
    previous: float,
    as_percentage: bool = True,
) -> Optional[float]:
    """
    Calculate the change between two values.
    
    Args:
        current: Current value.
        previous: Previous value.
        as_percentage: Return as percentage change.
        
    Returns:
        Change value, or None if previous is zero.
    """
    if previous == 0:
        return None
    
    change = current - previous
    
    if as_percentage:
        return (change / abs(previous)) * 100
    
    return change


def format_change(
    change: Optional[float],
    positive_prefix: str = "+",
    negative_prefix: str = "",
    decimals: int = 1,
) -> str:
    """
    Format a change value with appropriate prefix.
    
    Args:
        change: The change value (can be None).
        positive_prefix: Prefix for positive changes.
        negative_prefix: Prefix for negative changes.
        decimals: Decimal places.
        
    Returns:
        Formatted change string.
        
    Example:
        >>> format_change(5.5)
        '+5.5%'
        >>> format_change(-3.2)
        '-3.2%'
    """
    if change is None:
        return "N/A"
    
    prefix = positive_prefix if change >= 0 else negative_prefix
    return f"{prefix}{change:.{decimals}f}%"


def compare_periods(
    current: Dict[str, Any],
    previous: Dict[str, Any],
    metrics_to_compare: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics between two periods.
    
    Args:
        current: Current period metrics.
        previous: Previous period metrics.
        metrics_to_compare: Specific metrics to compare (defaults to all).
        
    Returns:
        Dictionary with current, previous, change, and formatted change.
        
    Example:
        >>> current = {"CTR": 0.05, "CVR": 0.03}
        >>> previous = {"CTR": 0.04, "CVR": 0.035}
        >>> compare_periods(current, previous)
        {
            "CTR": {"current": 0.05, "previous": 0.04, "change": 25.0, "change_formatted": "+25.0%"},
            "CVR": {"current": 0.03, "previous": 0.035, "change": -14.3, "change_formatted": "-14.3%"}
        }
    """
    keys = metrics_to_compare or list(set(current.keys()) & set(previous.keys()))
    
    comparison: Dict[str, Dict[str, Any]] = {}
    
    for key in keys:
        curr_val = current.get(key)
        prev_val = previous.get(key)
        
        if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
            change = calculate_change(curr_val, prev_val)
            comparison[key] = {
                "current": curr_val,
                "previous": prev_val,
                "change": round(change, 2) if change is not None else None,
                "change_formatted": format_change(change),
            }
        else:
            comparison[key] = {
                "current": curr_val,
                "previous": prev_val,
                "change": None,
                "change_formatted": "N/A",
            }
    
    return comparison


# ============== EXPORTS ==============
__all__ = [
    # Main formatting
    "format_metrics",
    "format_metrics_raw",
    "format_value",
    
    # Individual formatters
    "format_percentage",
    "format_currency",
    "format_integer",
    "format_float",
    "format_duration",
    
    # Report generation
    "format_as_table",
    "format_as_markdown_table",
    "generate_summary_report",
    
    # Export
    "export_to_csv",
    "export_to_json",
    
    # Comparison
    "calculate_change",
    "format_change",
    "compare_periods",
    
    # Configuration
    "FormatConfig",
    "MetricType",
    "detect_metric_type",
]

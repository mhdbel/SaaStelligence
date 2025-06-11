# utils/report_utils.py

def format_metrics(metrics):
    formatted = {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()}
    return formatted
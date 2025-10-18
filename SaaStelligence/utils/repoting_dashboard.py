"""Google Sheets dashboard integration helpers."""

from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - optional dependency
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:  # pragma: no cover - optional dependency
    gspread = None
    ServiceAccountCredentials = None


def update_dashboard(
    report: Dict[str, float],
    credentials_path: Optional[Path] = None,
    sheet_name: str = "SAAStelligence Dashboard",
) -> bool:
    """Push the latest metrics to Google Sheets when credentials are available."""

    if gspread is None or ServiceAccountCredentials is None:
        return False

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    creds_path = Path(credentials_path or "credentials.json")
    if not creds_path.exists():
        return False

    credentials = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), scope)
    client = gspread.authorize(credentials)
    sheet = client.open(sheet_name).sheet1
    sheet.update_row(2, list(report.values()))
    return True

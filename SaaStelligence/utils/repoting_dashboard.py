"""
Google Sheets dashboard integration for SAAStelligence metrics.

This module provides functionality to push performance metrics to a
Google Sheets dashboard for real-time monitoring and reporting.

Requirements:
    pip install gspread google-auth google-auth-oauthlib

Setup:
    1. Create a Google Cloud project
    2. Enable the Google Sheets API
    3. Create a service account and download credentials JSON
    4. Share your Google Sheet with the service account email

Example:
    >>> from utils.reporting_dashboard import DashboardClient
    >>> 
    >>> client = DashboardClient(credentials_path="credentials.json")
    >>> metrics = {"CTR": 0.0234, "CVR": 0.0512, "CPA": 45.67, "Leads": 150}
    >>> success = client.update_metrics(metrics)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# ============== LOGGING ==============
logger = logging.getLogger(__name__)

# ============== DEPENDENCY HANDLING ==============
# Try modern google-auth first, fall back to legacy oauth2client
GSPREAD_AVAILABLE = False
AUTH_METHOD = None

try:
    import gspread
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    GSPREAD_AVAILABLE = True
    AUTH_METHOD = "google-auth"
    logger.debug("Using google-auth for authentication")
except ImportError:
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        GSPREAD_AVAILABLE = True
        AUTH_METHOD = "oauth2client"
        logger.debug("Using oauth2client for authentication (deprecated)")
    except ImportError:
        gspread = None  # type: ignore
        ServiceAccountCredentials = None  # type: ignore
        logger.debug("Google Sheets integration not available - install gspread and google-auth")


# ============== TYPE DEFINITIONS ==============
T = TypeVar('T')
MetricsDict = Dict[str, Union[float, int, str]]


# ============== EXCEPTIONS ==============
class DashboardError(Exception):
    """Base exception for dashboard errors."""
    pass


class DashboardAuthError(DashboardError):
    """Raised when authentication fails."""
    pass


class DashboardConnectionError(DashboardError):
    """Raised when connection to Google Sheets fails."""
    pass


class DashboardUpdateError(DashboardError):
    """Raised when updating the sheet fails."""
    pass


# ============== CONFIGURATION ==============
@dataclass
class DashboardConfig:
    """Configuration for the dashboard client."""
    
    # Sheet settings
    sheet_name: str = "SAAStelligence Dashboard"
    worksheet_index: int = 0  # First worksheet
    header_row: int = 1
    data_start_row: int = 2
    
    # Behavior settings
    create_headers: bool = True
    append_mode: bool = False  # If True, append new rows; if False, update row
    include_timestamp: bool = True
    timestamp_column: str = "Updated At"
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    retry_backoff: float = 2.0  # multiplier
    
    # Rate limiting
    min_update_interval: float = 1.0  # seconds between updates
    
    # Column order (if None, uses dict key order)
    column_order: Optional[List[str]] = None


DEFAULT_CONFIG = DashboardConfig()

# Google Sheets API scopes
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ============== RETRY DECORATOR ==============
def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exceptions to catch and retry.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
            
            raise last_exception  # type: ignore
        
        return wrapper
    return decorator


# ============== DASHBOARD CLIENT ==============
class DashboardClient:
    """
    Client for updating Google Sheets dashboard with metrics.
    
    This client handles authentication, connection management, and
    provides robust error handling for dashboard updates.
    
    Example:
        >>> client = DashboardClient(
        ...     credentials_path="path/to/credentials.json",
        ...     config=DashboardConfig(sheet_name="My Dashboard")
        ... )
        >>> 
        >>> if client.is_available:
        ...     success = client.update_metrics({"CTR": 0.05, "CVR": 0.03})
        ...     print(f"Update {'succeeded' if success else 'failed'}")
    """
    
    def __init__(
        self,
        credentials_path: Optional[Union[str, Path]] = None,
        config: Optional[DashboardConfig] = None,
    ) -> None:
        """
        Initialize the dashboard client.
        
        Args:
            credentials_path: Path to Google service account credentials JSON.
            config: Dashboard configuration options.
        """
        self.config = config or DEFAULT_CONFIG
        self._credentials_path = Path(credentials_path) if credentials_path else None
        self._client: Optional[Any] = None
        self._sheet: Optional[Any] = None
        self._worksheet: Optional[Any] = None
        self._last_update: float = 0
        self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if the dashboard integration is available."""
        if not GSPREAD_AVAILABLE:
            return False
        
        if self._credentials_path is None:
            # Try default path
            default_path = Path("credentials.json")
            if default_path.exists():
                self._credentials_path = default_path
            else:
                return False
        
        return self._credentials_path.exists()
    
    @property
    def auth_method(self) -> Optional[str]:
        """Get the authentication method being used."""
        return AUTH_METHOD
    
    def _get_credentials(self) -> Any:
        """
        Load credentials from the JSON file.
        
        Returns:
            Credentials object for Google API authentication.
            
        Raises:
            DashboardAuthError: If credentials cannot be loaded.
        """
        if self._credentials_path is None or not self._credentials_path.exists():
            raise DashboardAuthError(
                f"Credentials file not found: {self._credentials_path}"
            )
        
        try:
            if AUTH_METHOD == "google-auth":
                return ServiceAccountCredentials.from_service_account_file(
                    str(self._credentials_path),
                    scopes=SCOPES,
                )
            else:
                # Legacy oauth2client
                return ServiceAccountCredentials.from_json_keyfile_name(
                    str(self._credentials_path),
                    SCOPES,
                )
        except Exception as e:
            raise DashboardAuthError(f"Failed to load credentials: {e}")
    
    def _ensure_connected(self) -> None:
        """
        Ensure connection to Google Sheets is established.
        
        Raises:
            DashboardConnectionError: If connection fails.
        """
        if self._worksheet is not None:
            return
        
        if not self.is_available:
            raise DashboardConnectionError(
                "Dashboard not available - check credentials path"
            )
        
        try:
            credentials = self._get_credentials()
            
            if AUTH_METHOD == "google-auth":
                self._client = gspread.authorize(credentials)
            else:
                self._client = gspread.authorize(credentials)
            
            self._sheet = self._client.open(self.config.sheet_name)
            self._worksheet = self._sheet.get_worksheet(self.config.worksheet_index)
            
            if self._worksheet is None:
                self._worksheet = self._sheet.sheet1
            
            self._initialized = True
            logger.info(f"Connected to dashboard: {self.config.sheet_name}")
            
        except gspread.SpreadsheetNotFound:
            raise DashboardConnectionError(
                f"Spreadsheet not found: {self.config.sheet_name}. "
                f"Make sure the sheet exists and is shared with the service account."
            )
        except gspread.APIError as e:
            raise DashboardConnectionError(f"Google Sheets API error: {e}")
        except Exception as e:
            raise DashboardConnectionError(f"Failed to connect to dashboard: {e}")
    
    def _respect_rate_limit(self) -> None:
        """Ensure minimum interval between updates."""
        elapsed = time.time() - self._last_update
        if elapsed < self.config.min_update_interval:
            sleep_time = self.config.min_update_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def _prepare_row_data(
        self,
        metrics: MetricsDict,
        include_timestamp: bool = True,
    ) -> tuple[List[str], List[Any]]:
        """
        Prepare header and data rows from metrics dictionary.
        
        Args:
            metrics: Dictionary of metric names to values.
            include_timestamp: Whether to include a timestamp column.
            
        Returns:
            Tuple of (headers, values) lists.
        """
        # Determine column order
        if self.config.column_order:
            keys = [k for k in self.config.column_order if k in metrics]
            # Add any keys not in the order
            keys.extend(k for k in metrics.keys() if k not in keys)
        else:
            keys = list(metrics.keys())
        
        headers = list(keys)
        values = [metrics.get(k, "") for k in keys]
        
        # Add timestamp if requested
        if include_timestamp and self.config.include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            headers.append(self.config.timestamp_column)
            values.append(timestamp)
        
        # Convert all values to strings/numbers that Sheets can handle
        processed_values = []
        for v in values:
            if v is None:
                processed_values.append("")
            elif isinstance(v, (int, float)):
                processed_values.append(v)
            else:
                processed_values.append(str(v))
        
        return headers, processed_values
    
    @with_retry(max_retries=3, delay=1.0, backoff=2.0)
    def _update_headers(self, headers: List[str]) -> None:
        """Update the header row in the sheet."""
        self._worksheet.update(
            f"A{self.config.header_row}",
            [headers],
            value_input_option="RAW",
        )
        logger.debug(f"Updated headers: {headers}")
    
    @with_retry(max_retries=3, delay=1.0, backoff=2.0)
    def _update_data_row(self, row: int, values: List[Any]) -> None:
        """Update a specific data row in the sheet."""
        self._worksheet.update(
            f"A{row}",
            [values],
            value_input_option="USER_ENTERED",
        )
        logger.debug(f"Updated row {row}")
    
    @with_retry(max_retries=3, delay=1.0, backoff=2.0)
    def _append_row(self, values: List[Any]) -> None:
        """Append a new row to the sheet."""
        self._worksheet.append_row(
            values,
            value_input_option="USER_ENTERED",
        )
        logger.debug("Appended new row")
    
    def update_metrics(
        self,
        metrics: MetricsDict,
        row: Optional[int] = None,
        update_headers: Optional[bool] = None,
    ) -> bool:
        """
        Update the dashboard with new metrics.
        
        Args:
            metrics: Dictionary of metric names to values.
            row: Specific row to update (defaults to config.data_start_row).
            update_headers: Whether to update headers (defaults to config.create_headers).
            
        Returns:
            True if update succeeded, False otherwise.
            
        Example:
            >>> client.update_metrics({
            ...     "CTR": 0.0234,
            ...     "CVR": 0.0512,
            ...     "CPA": 45.67,
            ...     "Leads": 150
            ... })
            True
        """
        if not self.is_available:
            logger.warning("Dashboard not available - skipping update")
            return False
        
        try:
            self._ensure_connected()
            self._respect_rate_limit()
            
            headers, values = self._prepare_row_data(metrics)
            
            # Update headers if requested
            should_update_headers = (
                update_headers if update_headers is not None 
                else self.config.create_headers
            )
            
            if should_update_headers:
                self._update_headers(headers)
            
            # Update or append data
            if self.config.append_mode:
                self._append_row(values)
            else:
                target_row = row or self.config.data_start_row
                self._update_data_row(target_row, values)
            
            self._last_update = time.time()
            logger.info(
                f"Dashboard updated successfully with {len(metrics)} metrics"
            )
            return True
            
        except DashboardError as e:
            logger.error(f"Dashboard update failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating dashboard: {e}", exc_info=True)
            return False
    
    def get_current_metrics(self) -> Optional[MetricsDict]:
        """
        Retrieve current metrics from the dashboard.
        
        Returns:
            Dictionary of current metrics, or None if retrieval fails.
        """
        if not self.is_available:
            return None
        
        try:
            self._ensure_connected()
            
            # Get headers and data
            headers = self._worksheet.row_values(self.config.header_row)
            values = self._worksheet.row_values(self.config.data_start_row)
            
            if not headers or not values:
                return None
            
            # Combine into dictionary
            metrics: MetricsDict = {}
            for header, value in zip(headers, values):
                if header and header != self.config.timestamp_column:
                    # Try to convert to number
                    try:
                        if "." in str(value):
                            metrics[header] = float(value)
                        else:
                            metrics[header] = int(value)
                    except (ValueError, TypeError):
                        metrics[header] = value
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return None
    
    def clear_data(self, keep_headers: bool = True) -> bool:
        """
        Clear data from the dashboard.
        
        Args:
            keep_headers: Whether to preserve the header row.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_available:
            return False
        
        try:
            self._ensure_connected()
            
            start_row = (
                self.config.data_start_row if keep_headers 
                else self.config.header_row
            )
            
            # Get the range to clear
            row_count = self._worksheet.row_count
            col_count = self._worksheet.col_count
            
            if row_count > start_row:
                range_to_clear = f"A{start_row}:{chr(64 + min(col_count, 26))}{row_count}"
                self._worksheet.batch_clear([range_to_clear])
                logger.info(f"Cleared dashboard data (rows {start_row}-{row_count})")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear dashboard: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Google Sheets."""
        self._client = None
        self._sheet = None
        self._worksheet = None
        self._initialized = False
        logger.debug("Disconnected from dashboard")


# ============== CONVENIENCE FUNCTION ==============
# Maintains backward compatibility with original API

_default_client: Optional[DashboardClient] = None


def update_dashboard(
    report: MetricsDict,
    credentials_path: Optional[Path] = None,
    sheet_name: str = "SAAStelligence Dashboard",
) -> bool:
    """
    Push the latest metrics to Google Sheets.
    
    This is a convenience function that maintains backward compatibility
    with the original API. For more control, use DashboardClient directly.
    
    Args:
        report: Dictionary of metric names to values.
        credentials_path: Path to Google service account credentials.
        sheet_name: Name of the Google Sheet to update.
        
    Returns:
        True if update succeeded, False otherwise.
        
    Example:
        >>> from utils.reporting_dashboard import update_dashboard
        >>> 
        >>> metrics = {"CTR": 0.0234, "CVR": 0.0512, "CPA": 45.67}
        >>> success = update_dashboard(metrics, credentials_path=Path("creds.json"))
    """
    global _default_client
    
    # Check if dependencies are available
    if not GSPREAD_AVAILABLE:
        logger.debug(
            "Google Sheets integration not available. "
            "Install with: pip install gspread google-auth"
        )
        return False
    
    # Create or reconfigure client
    if _default_client is None or sheet_name != _default_client.config.sheet_name:
        config = DashboardConfig(sheet_name=sheet_name)
        _default_client = DashboardClient(
            credentials_path=credentials_path,
            config=config,
        )
    elif credentials_path and credentials_path != _default_client._credentials_path:
        _default_client._credentials_path = Path(credentials_path)
        _default_client.disconnect()
    
    return _default_client.update_metrics(report)


def is_dashboard_available(credentials_path: Optional[Path] = None) -> bool:
    """
    Check if the dashboard integration is available and configured.
    
    Args:
        credentials_path: Path to credentials file to check.
        
    Returns:
        True if dashboard can be used, False otherwise.
    """
    if not GSPREAD_AVAILABLE:
        return False
    
    if credentials_path:
        return Path(credentials_path).exists()
    
    # Check default path
    return Path("credentials.json").exists()


# ============== EXPORTS ==============
__all__ = [
    # Main client
    "DashboardClient",
    "DashboardConfig",
    
    # Convenience functions
    "update_dashboard",
    "is_dashboard_available",
    
    # Exceptions
    "DashboardError",
    "DashboardAuthError",
    "DashboardConnectionError",
    "DashboardUpdateError",
    
    # Status
    "GSPREAD_AVAILABLE",
    "AUTH_METHOD",
]

"""
Configuration management for the SAAStelligence Engine.

This module provides centralized configuration with:
- Environment variable loading with validation
- Type-safe settings with Pydantic
- Environment-specific configurations (dev/staging/prod)
- Secrets protection
- Default value management

Usage:
    from config.config import CONFIG
    
    # Access settings
    api_key = CONFIG.OPENAI_API_KEY
    debug_mode = CONFIG.DEBUG
    
    # Check if in production
    if CONFIG.is_production:
        # Production-specific logic
        pass

Environment Variables:
    See .env.example for all available settings.
"""

from __future__ import annotations

import logging
import os
import secrets
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional, Set
from urllib.parse import urlparse

# Optional Pydantic import for enhanced validation
try:
    from pydantic import field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseSettings = object
    SettingsConfigDict = None

# Optional dotenv import
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs) -> bool:
        return False

# ============== INITIALIZATION ==============
# Load environment variables from .env file
_env_loaded = load_dotenv()

# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parents[1]

# ============== LOGGING ==============
logger = logging.getLogger(__name__)


# ============== ENVIRONMENT DETECTION ==============
class Environment:
    """Environment type constants."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    
    ALL = {DEVELOPMENT, STAGING, PRODUCTION, TESTING}
    
    @classmethod
    def from_string(cls, value: str) -> str:
        """Parse environment string with validation."""
        normalized = value.lower().strip()
        
        # Handle common aliases
        aliases = {
            "dev": cls.DEVELOPMENT,
            "develop": cls.DEVELOPMENT,
            "stage": cls.STAGING,
            "stg": cls.STAGING,
            "prod": cls.PRODUCTION,
            "prd": cls.PRODUCTION,
            "test": cls.TESTING,
        }
        
        resolved = aliases.get(normalized, normalized)
        
        if resolved not in cls.ALL:
            logger.warning(
                f"Unknown environment '{value}', defaulting to {cls.DEVELOPMENT}"
            )
            return cls.DEVELOPMENT
        
        return resolved


# ============== HELPER FUNCTIONS ==============
def _get_env(
    key: str,
    default: Any = None,
    required: bool = False,
    cast: type = str,
) -> Any:
    """
    Get environment variable with type casting and validation.
    
    Args:
        key: Environment variable name.
        default: Default value if not set.
        required: If True, raise error when not set.
        cast: Type to cast the value to.
        
    Returns:
        The environment variable value, cast to the specified type.
        
    Raises:
        ValueError: If required variable is not set.
        TypeError: If casting fails.
    """
    value = os.getenv(key)
    
    if value is None:
        if required:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Please set it in your environment or .env file."
            )
        return default
    
    # Handle boolean casting
    if cast is bool:
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    # Handle list casting (comma-separated)
    if cast is list:
        return [item.strip() for item in value.split(',') if item.strip()]
    
    # Handle int/float with validation
    if cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            logger.warning(f"Invalid {cast.__name__} value for {key}: {value}")
            return default
    
    return cast(value)


def _validate_url(url: str, name: str = "URL") -> str:
    """Validate URL format."""
    if not url:
        return url
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError(f"Invalid {name}: {url}")
        return url
    except Exception as e:
        logger.warning(f"Invalid {name} '{url}': {e}")
        return url


def _validate_path(path: str, must_exist: bool = False, name: str = "Path") -> Path:
    """Validate and convert path string to Path object."""
    p = Path(path)
    
    if must_exist and not p.exists():
        logger.warning(f"{name} does not exist: {path}")
    
    return p


def _mask_secret(value: Optional[str], visible_chars: int = 4) -> str:
    """Mask a secret value for safe logging."""
    if not value:
        return "<not set>"
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "*" * (len(value) - visible_chars)


# ============== CONFIGURATION CLASS ==============
if PYDANTIC_AVAILABLE:
    # Use Pydantic for validation when available
    
    class Config(BaseSettings):
        """
        Application configuration with Pydantic validation.
        
        All settings can be overridden via environment variables.
        See .env.example for documentation of all available settings.
        """
        
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore",
        )
        
        # ===== Environment =====
        ENVIRONMENT: str = "development"
        DEBUG: bool = False
        TESTING: bool = False
        
        # ===== Server =====
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        WORKERS: int = 1
        
        # ===== Logging =====
        LOG_LEVEL: str = "INFO"
        LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # ===== Security =====
        SECRET_KEY: Optional[str] = None
        API_KEY: Optional[str] = None
        REQUIRE_API_KEY: bool = False
        ALLOWED_ORIGINS: str = "*"
        ALLOWED_HOSTS: str = "*"
        
        # ===== Rate Limiting =====
        RATE_LIMIT_ENABLED: bool = True
        RATE_LIMIT_DEFAULT: str = "100/minute"
        RATE_LIMIT_GENERATE: str = "30/minute"
        RATE_LIMIT_REPORT: str = "10/minute"
        
        # ===== Paths =====
        INTENT_MODEL_PATH: str = str(BASE_DIR / "models" / "intent_classifier.json")
        CONVERSIONS_DATA_PATH: str = str(BASE_DIR / "data" / "conversions.csv")
        
        # ===== OpenAI / LLM =====
        OPENAI_API_KEY: Optional[str] = None
        OPENAI_MODEL: str = "gpt-3.5-turbo"
        OPENAI_TEMPERATURE: float = 0.7
        OPENAI_MAX_TOKENS: int = 500
        OPENAI_TIMEOUT: int = 30
        
        # ===== External Integrations =====
        META_ACCESS_TOKEN: Optional[str] = None
        GOOGLE_ADS_CLIENT_ID: Optional[str] = None
        GOOGLE_CREDENTIALS_PATH: Optional[str] = None
        HUBSPOT_API_KEY: Optional[str] = None
        
        # ===== Agent Configuration =====
        BASE_BID: float = 10.0
        DEFAULT_CPA_BUDGET: float = 45.0
        MAX_QUERY_LENGTH: int = 1000
        MIN_CONFIDENCE_THRESHOLD: float = 0.3
        MODEL_CACHE_TTL: int = 3600
        AGENT_MAX_WORKERS: int = 4
        RETARGET_BASE_URL: str = "https://ads.example.com/retarget"
        
        # ===== Validators =====
        @field_validator('ENVIRONMENT')
        @classmethod
        def validate_environment(cls, v: str) -> str:
            return Environment.from_string(v)
        
        @field_validator('LOG_LEVEL')
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
            upper = v.upper()
            if upper not in valid_levels:
                logger.warning(f"Invalid LOG_LEVEL '{v}', using INFO")
                return 'INFO'
            return upper
        
        @field_validator('PORT')
        @classmethod
        def validate_port(cls, v: int) -> int:
            if not 1 <= v <= 65535:
                raise ValueError(f"PORT must be between 1 and 65535, got {v}")
            return v
        
        @field_validator('ALLOWED_ORIGINS')
        @classmethod
        def parse_allowed_origins(cls, v: str) -> str:
            # Keep as string, will be parsed to list when needed
            return v
        
        @model_validator(mode='after')
        def validate_security(self) -> 'Config':
            """Validate security settings based on environment."""
            if self.is_production:
                if self.DEBUG:
                    logger.warning("DEBUG is enabled in production!")
                
                if self.ALLOWED_ORIGINS == "*":
                    logger.warning(
                        "ALLOWED_ORIGINS is '*' in production - "
                        "consider restricting to specific domains"
                    )
                
                if not self.SECRET_KEY:
                    # Generate a secret key if not provided
                    self.SECRET_KEY = secrets.token_urlsafe(32)
                    logger.warning(
                        "SECRET_KEY not set in production - "
                        "generated ephemeral key (will change on restart)"
                    )
            
            return self
        
        # ===== Computed Properties =====
        @cached_property
        def is_production(self) -> bool:
            """Check if running in production mode."""
            return self.ENVIRONMENT == Environment.PRODUCTION
        
        @cached_property
        def is_development(self) -> bool:
            """Check if running in development mode."""
            return self.ENVIRONMENT == Environment.DEVELOPMENT
        
        @cached_property
        def is_testing(self) -> bool:
            """Check if running in testing mode."""
            return self.TESTING or self.ENVIRONMENT == Environment.TESTING
        
        @cached_property
        def allowed_origins_list(self) -> List[str]:
            """Get allowed origins as a list."""
            if self.ALLOWED_ORIGINS == "*":
                return ["*"]
            return [
                origin.strip() 
                for origin in self.ALLOWED_ORIGINS.split(",") 
                if origin.strip()
            ]
        
        @cached_property
        def intent_model_path(self) -> Path:
            """Get intent model path as Path object."""
            return Path(self.INTENT_MODEL_PATH)
        
        @cached_property
        def conversions_data_path(self) -> Path:
            """Get conversions data path as Path object."""
            return Path(self.CONVERSIONS_DATA_PATH)
        
        @cached_property
        def has_openai(self) -> bool:
            """Check if OpenAI is configured."""
            return bool(self.OPENAI_API_KEY)
        
        @cached_property
        def has_meta(self) -> bool:
            """Check if Meta/Facebook integration is configured."""
            return bool(self.META_ACCESS_TOKEN)
        
        @cached_property
        def has_google_ads(self) -> bool:
            """Check if Google Ads integration is configured."""
            return bool(self.GOOGLE_ADS_CLIENT_ID)
        
        @cached_property
        def has_hubspot(self) -> bool:
            """Check if HubSpot integration is configured."""
            return bool(self.HUBSPOT_API_KEY)
        
        # ===== Methods =====
        def get_masked_secrets(self) -> dict:
            """Get configuration with secrets masked for safe logging."""
            return {
                "ENVIRONMENT": self.ENVIRONMENT,
                "DEBUG": self.DEBUG,
                "HOST": self.HOST,
                "PORT": self.PORT,
                "LOG_LEVEL": self.LOG_LEVEL,
                "OPENAI_API_KEY": _mask_secret(self.OPENAI_API_KEY),
                "META_ACCESS_TOKEN": _mask_secret(self.META_ACCESS_TOKEN),
                "GOOGLE_ADS_CLIENT_ID": _mask_secret(self.GOOGLE_ADS_CLIENT_ID),
                "HUBSPOT_API_KEY": _mask_secret(self.HUBSPOT_API_KEY),
                "API_KEY": _mask_secret(self.API_KEY),
                "SECRET_KEY": _mask_secret(self.SECRET_KEY),
                "INTENT_MODEL_PATH": self.INTENT_MODEL_PATH,
                "CONVERSIONS_DATA_PATH": self.CONVERSIONS_DATA_PATH,
                "REQUIRE_API_KEY": self.REQUIRE_API_KEY,
                "ALLOWED_ORIGINS": self.ALLOWED_ORIGINS,
            }
        
        def validate_paths(self) -> dict:
            """Validate that required paths exist."""
            results = {}
            
            model_path = self.intent_model_path
            results['intent_model'] = {
                'path': str(model_path),
                'exists': model_path.exists(),
                'is_file': model_path.is_file() if model_path.exists() else False,
            }
            
            data_path = self.conversions_data_path
            results['conversions_data'] = {
                'path': str(data_path),
                'exists': data_path.exists(),
                'is_file': data_path.is_file() if data_path.exists() else False,
            }
            
            # Check parent directories
            results['model_dir'] = {
                'path': str(model_path.parent),
                'exists': model_path.parent.exists(),
                'writable': os.access(model_path.parent, os.W_OK) if model_path.parent.exists() else False,
            }
            
            results['data_dir'] = {
                'path': str(data_path.parent),
                'exists': data_path.parent.exists(),
                'writable': os.access(data_path.parent, os.W_OK) if data_path.parent.exists() else False,
            }
            
            return results
        
        def __repr__(self) -> str:
            return (
                f"Config(environment={self.ENVIRONMENT}, "
                f"debug={self.DEBUG}, "
                f"host={self.HOST}:{self.PORT})"
            )

else:
    # Fallback implementation without Pydantic
    
    class Config:
        """
        Application configuration (fallback without Pydantic).
        
        For better validation, install pydantic:
            pip install pydantic pydantic-settings
        """
        
        def __init__(self) -> None:
            # ===== Environment =====
            self.ENVIRONMENT = Environment.from_string(
                _get_env('ENVIRONMENT', 'development')
            )
            self.DEBUG = _get_env('DEBUG', False, cast=bool)
            self.TESTING = _get_env('TESTING', False, cast=bool)
            
            # ===== Server =====
            self.HOST = _get_env('HOST', '0.0.0.0')
            self.PORT = _get_env('PORT', 8000, cast=int)
            self.WORKERS = _get_env('WORKERS', 1, cast=int)
            
            # ===== Logging =====
            self.LOG_LEVEL = _get_env('LOG_LEVEL', 'INFO').upper()
            self.LOG_FORMAT = _get_env(
                'LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # ===== Security =====
            self.SECRET_KEY = _get_env('SECRET_KEY', None)
            self.API_KEY = _get_env('API_KEY', None)
            self.REQUIRE_API_KEY = _get_env('REQUIRE_API_KEY', False, cast=bool)
            self.ALLOWED_ORIGINS = _get_env('ALLOWED_ORIGINS', '*')
            self.ALLOWED_HOSTS = _get_env('ALLOWED_HOSTS', '*')
            
            # ===== Rate Limiting =====
            self.RATE_LIMIT_ENABLED = _get_env('RATE_LIMIT_ENABLED', True, cast=bool)
            self.RATE_LIMIT_DEFAULT = _get_env('RATE_LIMIT_DEFAULT', '100/minute')
            self.RATE_LIMIT_GENERATE = _get_env('RATE_LIMIT_GENERATE', '30/minute')
            self.RATE_LIMIT_REPORT = _get_env('RATE_LIMIT_REPORT', '10/minute')
            
            # ===== Paths =====
            self.INTENT_MODEL_PATH = _get_env(
                'INTENT_MODEL_PATH',
                str(BASE_DIR / 'models' / 'intent_classifier.json')
            )
            self.CONVERSIONS_DATA_PATH = _get_env(
                'CONVERSIONS_DATA_PATH',
                str(BASE_DIR / 'data' / 'conversions.csv')
            )
            
            # ===== OpenAI / LLM =====
            self.OPENAI_API_KEY = _get_env('OPENAI_API_KEY', None)
            self.OPENAI_MODEL = _get_env('OPENAI_MODEL', 'gpt-3.5-turbo')
            self.OPENAI_TEMPERATURE = _get_env('OPENAI_TEMPERATURE', 0.7, cast=float)
            self.OPENAI_MAX_TOKENS = _get_env('OPENAI_MAX_TOKENS', 500, cast=int)
            self.OPENAI_TIMEOUT = _get_env('OPENAI_TIMEOUT', 30, cast=int)
            
            # ===== External Integrations =====
            self.META_ACCESS_TOKEN = _get_env('META_ACCESS_TOKEN', None)
            self.GOOGLE_ADS_CLIENT_ID = _get_env('GOOGLE_ADS_CLIENT_ID', None)
            self.GOOGLE_CREDENTIALS_PATH = _get_env('GOOGLE_CREDENTIALS_PATH', None)
            self.HUBSPOT_API_KEY = _get_env('HUBSPOT_API_KEY', None)
            
            # ===== Agent Configuration =====
            self.BASE_BID = _get_env('BASE_BID', 10.0, cast=float)
            self.DEFAULT_CPA_BUDGET = _get_env('DEFAULT_CPA_BUDGET', 45.0, cast=float)
            self.MAX_QUERY_LENGTH = _get_env('MAX_QUERY_LENGTH', 1000, cast=int)
            self.MIN_CONFIDENCE_THRESHOLD = _get_env('MIN_CONFIDENCE_THRESHOLD', 0.3, cast=float)
            self.MODEL_CACHE_TTL = _get_env('MODEL_CACHE_TTL', 3600, cast=int)
            self.AGENT_MAX_WORKERS = _get_env('AGENT_MAX_WORKERS', 4, cast=int)
            self.RETARGET_BASE_URL = _get_env(
                'RETARGET_BASE_URL',
                'https://ads.example.com/retarget'
            )
            
            # Generate secret key for production if not set
            if self.is_production and not self.SECRET_KEY:
                self.SECRET_KEY = secrets.token_urlsafe(32)
                logger.warning("Generated ephemeral SECRET_KEY for production")
        
        @property
        def is_production(self) -> bool:
            """Check if running in production mode."""
            return self.ENVIRONMENT == Environment.PRODUCTION
        
        @property
        def is_development(self) -> bool:
            """Check if running in development mode."""
            return self.ENVIRONMENT == Environment.DEVELOPMENT
        
        @property
        def is_testing(self) -> bool:
            """Check if running in testing mode."""
            return self.TESTING or self.ENVIRONMENT == Environment.TESTING
        
        @property
        def allowed_origins_list(self) -> List[str]:
            """Get allowed origins as a list."""
            if self.ALLOWED_ORIGINS == "*":
                return ["*"]
            return [
                origin.strip()
                for origin in self.ALLOWED_ORIGINS.split(",")
                if origin.strip()
            ]
        
        @property
        def intent_model_path(self) -> Path:
            """Get intent model path as Path object."""
            return Path(self.INTENT_MODEL_PATH)
        
        @property
        def conversions_data_path(self) -> Path:
            """Get conversions data path as Path object."""
            return Path(self.CONVERSIONS_DATA_PATH)
        
        @property
        def has_openai(self) -> bool:
            """Check if OpenAI is configured."""
            return bool(self.OPENAI_API_KEY)
        
        @property
        def has_meta(self) -> bool:
            """Check if Meta/Facebook integration is configured."""
            return bool(self.META_ACCESS_TOKEN)
        
        @property
        def has_google_ads(self) -> bool:
            """Check if Google Ads integration is configured."""
            return bool(self.GOOGLE_ADS_CLIENT_ID)
        
        @property
        def has_hubspot(self) -> bool:
            """Check if HubSpot integration is configured."""
            return bool(self.HUBSPOT_API_KEY)
        
        def get_masked_secrets(self) -> dict:
            """Get configuration with secrets masked for safe logging."""
            return {
                "ENVIRONMENT": self.ENVIRONMENT,
                "DEBUG": self.DEBUG,
                "HOST": self.HOST,
                "PORT": self.PORT,
                "LOG_LEVEL": self.LOG_LEVEL,
                "OPENAI_API_KEY": _mask_secret(self.OPENAI_API_KEY),
                "META_ACCESS_TOKEN": _mask_secret(self.META_ACCESS_TOKEN),
                "GOOGLE_ADS_CLIENT_ID": _mask_secret(self.GOOGLE_ADS_CLIENT_ID),
                "HUBSPOT_API_KEY": _mask_secret(self.HUBSPOT_API_KEY),
                "API_KEY": _mask_secret(self.API_KEY),
                "SECRET_KEY": _mask_secret(self.SECRET_KEY),
                "INTENT_MODEL_PATH": self.INTENT_MODEL_PATH,
                "CONVERSIONS_DATA_PATH": self.CONVERSIONS_DATA_PATH,
                "REQUIRE_API_KEY": self.REQUIRE_API_KEY,
                "ALLOWED_ORIGINS": self.ALLOWED_ORIGINS,
            }
        
        def validate_paths(self) -> dict:
            """Validate that required paths exist."""
            results = {}
            
            model_path = self.intent_model_path
            results['intent_model'] = {
                'path': str(model_path),
                'exists': model_path.exists(),
                'is_file': model_path.is_file() if model_path.exists() else False,
            }
            
            data_path = self.conversions_data_path
            results['conversions_data'] = {
                'path': str(data_path),
                'exists': data_path.exists(),
                'is_file': data_path.is_file() if data_path.exists() else False,
            }
            
            return results
        
        def __repr__(self) -> str:
            return (
                f"Config(environment={self.ENVIRONMENT}, "
                f"debug={self.DEBUG}, "
                f"host={self.HOST}:{self.PORT})"
            )


# ============== SINGLETON INSTANCE ==============
CONFIG = Config()

# Log configuration on import (only in debug mode)
if CONFIG.DEBUG:
    logger.debug(f"Configuration loaded: {CONFIG.get_masked_secrets()}")


# ============== CONFIGURATION VALIDATION ==============
def validate_config() -> dict:
    """
    Validate the current configuration.
    
    Returns:
        Dictionary with validation results and any warnings/errors.
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'paths': CONFIG.validate_paths(),
        'integrations': {
            'openai': CONFIG.has_openai,
            'meta': CONFIG.has_meta,
            'google_ads': CONFIG.has_google_ads,
            'hubspot': CONFIG.has_hubspot,
        },
    }
    
    # Check paths
    if not CONFIG.intent_model_path.parent.exists():
        results['warnings'].append(
            f"Model directory does not exist: {CONFIG.intent_model_path.parent}"
        )
    
    if not CONFIG.conversions_data_path.parent.exists():
        results['warnings'].append(
            f"Data directory does not exist: {CONFIG.conversions_data_path.parent}"
        )
    
    # Check production requirements
    if CONFIG.is_production:
        if CONFIG.DEBUG:
            results['warnings'].append("DEBUG is enabled in production")
        
        if CONFIG.ALLOWED_ORIGINS == "*":
            results['warnings'].append("ALLOWED_ORIGINS is '*' in production")
        
        if not CONFIG.SECRET_KEY or len(CONFIG.SECRET_KEY) < 32:
            results['warnings'].append("SECRET_KEY should be at least 32 characters")
        
        if CONFIG.REQUIRE_API_KEY and not CONFIG.API_KEY:
            results['errors'].append("REQUIRE_API_KEY is true but API_KEY is not set")
            results['valid'] = False
    
    return results


# ============== EXPORTS ==============
__all__ = [
    'CONFIG',
    'Config',
    'Environment',
    'validate_config',
    'BASE_DIR',
]
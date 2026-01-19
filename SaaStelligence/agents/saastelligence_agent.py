"""
Core agent orchestration for the SAAStelligence pipeline.

This module provides the main SAAStelligenceAgent class that handles:
- Intent detection from user queries
- Dynamic ad copy generation
- Conversion funnel routing
- Bid adjustment based on predicted CVR
- Retargeting URL generation
- Performance reporting and analytics
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import quote
import re
import time

# Graceful LangChain import with version handling
try:
    # New import paths (langchain >= 0.1.0)
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
    LANGCHAIN_VERSION = ">=0.1.0"
except ImportError:
    try:
        # Legacy import paths (langchain < 0.1.0)
        from langchain import LLMChain
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        LANGCHAIN_AVAILABLE = True
        LANGCHAIN_VERSION = "<0.1.0"
    except ImportError:
        LLMChain = None
        OpenAI = None
        PromptTemplate = None
        LANGCHAIN_AVAILABLE = False
        LANGCHAIN_VERSION = None

# Internal imports - use relative imports
from config.config import CONFIG
from models.train_intent_model import (
    INTENT_CATEGORIES,
    REQUIRED_DATA_COLUMNS,
    load_model as load_intent_model,
    train_intent_classifier,
)
from utils.report_utils import format_metrics

# Optional dashboard import
try:
    from utils.reporting_dashboard import update_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    update_dashboard = None
    DASHBOARD_AVAILABLE = False

# ============== LOGGING ==============
logger = logging.getLogger(__name__)

# ============== CONSTANTS ==============
NUM_INTENTS = len(INTENT_CATEGORIES)
OPTIONAL_PERFORMANCE_COLUMNS = ["clicks", "impressions", "cost"]
CSV_FIELD_ORDER = ["query_text", "intent", "converted", *OPTIONAL_PERFORMANCE_COLUMNS]
MAX_CSV_RECORDS = 100_000  # Prevent OOM on large files


# ============== TYPE DEFINITIONS ==============
class PipelineResult(TypedDict):
    """Type definition for the complete pipeline output."""
    intent: str
    intent_confidence: float
    ad_copy: str
    funnel: str
    predicted_cvr: float
    bid: float
    retarget_url: Optional[str]


class PerformanceRecord(TypedDict):
    """Type definition for performance data records."""
    query_text: str
    intent: str
    converted: int
    clicks: float
    impressions: float
    cost: float


class MetricsResult(TypedDict):
    """Type definition for calculated metrics."""
    CTR: float
    CVR: float
    CPA: float
    Leads: int


# ============== EXCEPTIONS ==============
class AgentError(Exception):
    """
    Base exception for agent errors.
    
    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional context.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ModelNotFoundError(AgentError):
    """Raised when intent model cannot be loaded or trained."""
    pass


class ModelLoadError(AgentError):
    """Raised when model loading fails due to corruption or incompatibility."""
    pass


class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass


class AdGenerationError(AgentError):
    """Raised when ad generation fails after retries."""
    pass


# ============== CONFIGURATION ==============
@dataclass(frozen=True)
class AgentConfig:
    """
    Immutable configuration for the SAAStelligence agent.
    
    All values have sensible defaults and can be overridden
    via the CONFIG object or constructor parameters.
    """
    
    # Bidding Configuration
    base_bid: float = 10.0
    default_cpa_budget: float = 45.0
    low_cvr_multiplier: float = 0.6
    high_performance_multiplier: float = 1.3
    medium_performance_multiplier: float = 1.15
    low_performance_multiplier: float = 0.7
    below_target_multiplier: float = 0.9
    
    # CVR Thresholds
    min_target_cvr: float = 0.02
    max_target_cvr: float = 0.15
    default_cvr: float = 0.03
    
    # Performance Thresholds
    high_performance_ratio: float = 1.2
    low_performance_ratio: float = 0.6
    
    # Validation
    max_query_length: int = 1000
    min_confidence_threshold: float = 0.3
    
    # URLs
    retarget_base_url: str = "https://ads.example.com/retarget"
    
    # Model Cache
    model_cache_ttl_seconds: int = 3600  # 1 hour
    
    # Thread Pool
    max_workers: int = 4
    
    # Ad Generation
    ad_generation_max_retries: int = 3
    ad_generation_timeout: int = 30
    
    @classmethod
    def from_config(cls, config: Any) -> "AgentConfig":
        """
        Create AgentConfig from external CONFIG object.
        
        Uses defaults for any missing configuration values.
        """
        def get_config_value(name: str, default: Any) -> Any:
            return getattr(config, name, default)
        
        return cls(
            base_bid=get_config_value('BASE_BID', cls.base_bid),
            default_cpa_budget=get_config_value('DEFAULT_CPA_BUDGET', cls.default_cpa_budget),
            retarget_base_url=get_config_value('RETARGET_BASE_URL', cls.retarget_base_url),
            max_query_length=get_config_value('MAX_QUERY_LENGTH', cls.max_query_length),
            min_confidence_threshold=get_config_value('MIN_CONFIDENCE_THRESHOLD', cls.min_confidence_threshold),
            model_cache_ttl_seconds=get_config_value('MODEL_CACHE_TTL', cls.model_cache_ttl_seconds),
            max_workers=get_config_value('AGENT_MAX_WORKERS', cls.max_workers),
        )


# ============== CACHED MODEL WRAPPER ==============
@dataclass
class CachedModel:
    """
    Wrapper for cached model with TTL and validation.
    
    Tracks when the model was loaded and the file hash
    to detect when reloading is necessary.
    """
    model: Any
    loaded_at: datetime
    model_hash: str
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache has expired based on TTL."""
        return datetime.now() - self.loaded_at > timedelta(seconds=ttl_seconds)
    
    def is_stale(self, current_hash: str) -> bool:
        """Check if the underlying model file has changed."""
        return self.model_hash != current_hash


# ============== HELPER FUNCTIONS ==============
def _hash_query(query: str) -> str:
    """
    Create a truncated hash of query for safe logging.
    
    This prevents PII from appearing in logs while still
    allowing request correlation.
    """
    return hashlib.sha256(query.encode()).hexdigest()[:8]


def _sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input to prevent CSV injection and limit length.
    
    Removes dangerous leading characters that could trigger
    formula injection in spreadsheet applications.
    """
    if not value:
        return ""
    
    # Characters that could trigger formula injection
    dangerous_chars = ('=', '+', '-', '@', '\t', '\r', '\n', '|')
    
    sanitized = value.strip()
    while sanitized and sanitized[0] in dangerous_chars:
        sanitized = sanitized[1:]
    
    return sanitized[:max_length]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float with fallback."""
    if value in (None, ""):
        return default
    try:
        result = float(value)
        # Guard against infinity and NaN
        if not (-1e15 < result < 1e15):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _normalize_record(record: Dict[str, str]) -> PerformanceRecord:
    """
    Normalize a raw CSV record to consistent types.
    
    Includes sanitization to prevent injection attacks
    and type coercion for numeric fields.
    """
    return PerformanceRecord(
        query_text=_sanitize_string(str(record.get("query_text", ""))),
        intent=_sanitize_string(str(record.get("intent", "")), max_length=100),
        converted=int(_safe_float(record.get("converted", 0)) > 0),
        clicks=_safe_float(record.get("clicks")),
        impressions=_safe_float(record.get("impressions")),
        cost=_safe_float(record.get("cost")),
    )


# ============== MAIN AGENT CLASS ==============
class SAAStelligenceAgent:
    """
    Core AI agent for the SAAStelligence pipeline.
    
    This agent provides a complete pipeline for SaaS lead generation:
    
    1. **Intent Detection**: Classify user queries into intent categories
       using a trained ML model.
    
    2. **Ad Generation**: Create dynamic, personalized ad copy using
       LangChain and OpenAI (optional, requires API key).
    
    3. **Funnel Routing**: Route users to appropriate conversion funnels
       based on detected intent.
    
    4. **Bid Adjustment**: Calculate optimal ad bids based on predicted
       conversion rates and CPA budgets.
    
    5. **Retargeting**: Generate retargeting URLs for users based on
       their previous actions.
    
    6. **Performance Reporting**: Calculate and report aggregate metrics
       with optional dashboard integration.
    
    Thread Safety:
        This class is fully thread-safe. Model loading and data writes
        are protected by locks. For async contexts, use the `run_async()`
        and `report_performance_async()` methods.
    
    Example:
        >>> agent = SAAStelligenceAgent()
        >>> result = agent.run("I need help automating my sales workflow")
        >>> print(result['intent'])
        'workflow_automation'
        >>> print(result['ad_copy'])
        'Supercharge your sales with AI-powered automation...'
    
    Async Example:
        >>> agent = SAAStelligenceAgent()
        >>> result = await agent.run_async("I need CRM integration")
        >>> print(result['funnel'])
        'funnel_b'
    """
    
    # Validation patterns
    USER_ID_PATTERN: re.Pattern = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    
    # Default Funnel Mapping
    DEFAULT_FUNNEL_MAP: Dict[str, str] = {
        "workflow_automation": "funnel_a",
        "sales_team_efficiency": "funnel_b",
        "project_management": "funnel_c",
        "customer_support": "funnel_d",
        "marketing_automation": "funnel_e",
    }
    DEFAULT_FUNNEL: str = "default_funnel"

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """
        Initialize the SAAStelligence agent.
        
        Args:
            config: Optional configuration override. If not provided,
                    configuration is loaded from the global CONFIG object.
            executor: Optional thread pool executor for async operations.
                      If not provided, a new executor is created.
        """
        self._config = config or AgentConfig.from_config(CONFIG)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="saastelligence-"
        )
        self._owns_executor = executor is None
        
        # Intent mapping
        self.intent_mapping = {
            intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)
        }
        
        # Model cache with thread safety
        self._cached_model: Optional[CachedModel] = None
        self._model_lock = threading.RLock()
        self._data_lock = threading.RLock()
        
        # Paths
        self._model_path = Path(CONFIG.INTENT_MODEL_PATH)
        self._data_path = Path(CONFIG.CONVERSIONS_DATA_PATH)
        
        # Initialize ad generation chain
        self._ad_chain = self._build_ad_chain()
        
        logger.info(
            "SAAStelligenceAgent initialized",
            extra={
                "model_path": str(self._model_path),
                "data_path": str(self._data_path),
                "langchain_available": LANGCHAIN_AVAILABLE,
                "langchain_version": LANGCHAIN_VERSION,
                "ad_generation_enabled": self._ad_chain is not None,
            }
        )

    def __del__(self) -> None:
        """Cleanup resources on deletion."""
        if hasattr(self, '_owns_executor') and self._owns_executor:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"SAAStelligenceAgent("
            f"model_path={self._model_path}, "
            f"ad_generation={'enabled' if self._ad_chain else 'disabled'})"
        )

    # ------------------------------------------------------------------
    # Model Management
    # ------------------------------------------------------------------
    
    def _build_ad_chain(self) -> Optional[Any]:
        """
        Build LangChain ad generation chain if dependencies are available.
        
        Returns None if LangChain is not installed or OpenAI API key
        is not configured, allowing the agent to operate in a degraded
        mode without ad generation.
        """
        if not LANGCHAIN_AVAILABLE:
            logger.warning(
                "LangChain not available - ad generation disabled. "
                "Install with: pip install langchain langchain-openai"
            )
            return None
        
        api_key = getattr(CONFIG, 'OPENAI_API_KEY', None)
        if not api_key:
            logger.warning(
                "OpenAI API key not configured - ad generation disabled. "
                "Set OPENAI_API_KEY environment variable to enable."
            )
            return None

        try:
            ad_prompt_template = PromptTemplate.from_template(
                """You are an expert SaaS marketing copywriter. Based on this user intent: {intent}

Generate a high-conversion ad copy for SaaS lead generation that:
1. Addresses the specific pain point implied by the intent
2. Is emotionally engaging and creates urgency
3. Includes a clear call-to-action
4. Is concise (under 150 words)

Output only the ad text, no explanations."""
            )
            
            llm = OpenAI(
                openai_api_key=api_key,
                temperature=0.7,
                request_timeout=self._config.ad_generation_timeout,
                max_retries=self._config.ad_generation_max_retries,
            )
            
            chain = LLMChain(llm=llm, prompt=ad_prompt_template)
            
            logger.info("Ad generation chain configured successfully")
            return chain
            
        except Exception as e:
            logger.error(f"Failed to initialize ad generation chain: {e}")
            return None

    def _get_model_hash(self) -> str:
        """
        Get hash of model file for cache validation.
        
        Uses file modification time and size as a proxy for content hash,
        which is faster than reading the entire file.
        """
        if not self._model_path.exists():
            return ""
        try:
            stat = self._model_path.stat()
            return f"{stat.st_mtime:.6f}_{stat.st_size}"
        except OSError:
            return ""

    def _ensure_model(self) -> Any:
        """
        Ensure intent model is loaded, training if necessary.
        
        This method is thread-safe and implements caching with TTL.
        The cache is invalidated if:
        - The TTL has expired
        - The underlying model file has changed
        
        Returns:
            The loaded intent classification model.
            
        Raises:
            ModelNotFoundError: If model cannot be loaded and no training data exists.
            ModelLoadError: If model file is corrupted or incompatible.
        """
        with self._model_lock:
            current_hash = self._get_model_hash()
            
            # Check if cached model is still valid
            if self._cached_model is not None:
                cache = self._cached_model
                if (not cache.is_expired(self._config.model_cache_ttl_seconds) 
                    and not cache.is_stale(current_hash)):
                    return cache.model
                
                reason = "expired" if cache.is_expired(self._config.model_cache_ttl_seconds) else "stale"
                logger.info(f"Model cache {reason}, reloading")
            
            # Try to load existing model
            if self._model_path.exists():
                try:
                    logger.info(f"Loading intent model from {self._model_path}")
                    model = load_intent_model(self._model_path)
                    
                    # Validate model interface
                    if not hasattr(model, 'predict_proba'):
                        raise ModelLoadError(
                            "Invalid model: missing predict_proba method",
                            {"model_type": type(model).__name__}
                        )
                    
                    # Cache the model
                    self._cached_model = CachedModel(
                        model=model,
                        loaded_at=datetime.now(),
                        model_hash=current_hash,
                    )
                    
                    logger.info("Intent model loaded successfully")
                    return model
                    
                except ModelLoadError:
                    raise
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    raise ModelLoadError(
                        f"Failed to load model from {self._model_path}",
                        {"original_error": str(e)}
                    )
            
            # Attempt to train a new model
            logger.warning(
                f"Model not found at {self._model_path}, attempting to train from data"
            )
            
            dataset = self._load_performance_data()
            
            if not dataset:
                raise ModelNotFoundError(
                    f"Intent model not found and no training data available",
                    {
                        "model_path": str(self._model_path),
                        "data_path": str(self._data_path),
                        "solution": "Run the training script or provide conversion data",
                    }
                )
            
            # Train new model
            logger.info(f"Training new model with {len(dataset)} records")
            model = train_intent_classifier(
                records=[
                    {key: str(value) for key, value in record.items()}
                    for record in dataset
                ],
                model_path=self._model_path,
            )
            
            # Cache the newly trained model
            self._cached_model = CachedModel(
                model=model,
                loaded_at=datetime.now(),
                model_hash=self._get_model_hash(),
            )
            
            logger.info("Intent model trained and cached successfully")
            return model

    def invalidate_model_cache(self) -> None:
        """
        Explicitly invalidate the model cache.
        
        Call this after training or when you know the model
        file has been updated externally.
        """
        with self._model_lock:
            self._cached_model = None
            logger.info("Model cache invalidated")

    # ------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------
    
    def _validate_query(self, query: str) -> str:
        """
        Validate and sanitize query input.
        
        Args:
            query: Raw query string from user.
            
        Returns:
            Sanitized query string.
            
        Raises:
            ValidationError: If query is invalid.
        """
        if not isinstance(query, str):
            raise ValidationError(
                "Query must be a string",
                {"received_type": type(query).__name__}
            )
        
        query = query.strip()
        
        if not query:
            raise ValidationError("Query cannot be empty")
        
        if len(query) > self._config.max_query_length:
            raise ValidationError(
                f"Query exceeds maximum length of {self._config.max_query_length} characters",
                {"query_length": len(query), "max_length": self._config.max_query_length}
            )
        
        return query

    def _validate_user_id(self, user_id: str) -> bool:
        """
        Validate user ID format for security.
        
        Only allows alphanumeric characters, underscores, and hyphens.
        Maximum length of 64 characters.
        """
        return bool(self.USER_ID_PATTERN.match(user_id))

    # ------------------------------------------------------------------
    # Core Agent Capabilities
    # ------------------------------------------------------------------
    
    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect user intent from query text.
        
        Uses the trained intent classification model to predict
        the most likely intent category for the given query.
        
        Args:
            query: User's search query or input text.
            
        Returns:
            Tuple of (intent_label, confidence_score) where:
            - intent_label: One of the INTENT_CATEGORIES
            - confidence_score: Float between 0 and 1
            
        Raises:
            ValidationError: If query is invalid.
            ModelNotFoundError: If model cannot be loaded.
            
        Example:
            >>> intent, confidence = agent.detect_intent("automate my workflow")
            >>> print(f"{intent}: {confidence:.1%}")
            'workflow_automation: 87.3%'
        """
        query = self._validate_query(query)
        query_hash = _hash_query(query)
        
        logger.debug(f"Detecting intent for query (hash: {query_hash})")
        
        model = self._ensure_model()
        probabilities = model.predict_proba(query)
        
        intent_index = max(range(NUM_INTENTS), key=lambda idx: probabilities[idx])
        intent = INTENT_CATEGORIES[intent_index]
        confidence = float(probabilities[intent_index])
        
        # Log warning for low confidence predictions
        if confidence < self._config.min_confidence_threshold:
            logger.warning(
                f"Low confidence prediction: {confidence:.1%}",
                extra={"intent": intent, "query_hash": query_hash}
            )
        
        logger.info(
            f"Intent detected: {intent} (confidence: {confidence:.1%})",
            extra={"query_hash": query_hash}
        )
        
        return intent, confidence

    def generate_ad(self, intent: str) -> str:
        """
        Generate ad copy for the given intent.
        
        Uses LangChain with OpenAI to generate dynamic, personalized
        ad copy. Includes retry logic for transient failures.
        
        Args:
            intent: Detected user intent category.
            
        Returns:
            Generated ad copy string, or a fallback message if
            ad generation is not available.
            
        Example:
            >>> ad = agent.generate_ad("workflow_automation")
            >>> print(ad)
            'Transform your workflow with AI-powered automation...'
        """
        if self._ad_chain is None:
            return (
                "Dynamic ad generation is not configured. "
                "Set OPENAI_API_KEY environment variable and install "
                "langchain packages to enable AI-powered ad copy."
            )
        
        max_retries = self._config.ad_generation_max_retries
        last_error = None
        
        for attempt in range(max_retries):
            try:
                ad_copy = self._ad_chain.run(intent=intent)
                logger.debug(f"Generated ad copy for intent '{intent}'")
                return ad_copy.strip()
                
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"Ad generation attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s..."
                )
                
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
        
        logger.error(
            f"Ad generation failed after {max_retries} attempts",
            extra={"last_error": str(last_error), "intent": intent}
        )
        
        return (
            "We're experiencing high demand. Your personalized ad is being prepared. "
            "Please try again in a moment."
        )

    def route_to_funnel(self, intent: str) -> str:
        """
        Route user to appropriate conversion funnel based on intent.
        
        Args:
            intent: Detected user intent category.
            
        Returns:
            Funnel identifier string (e.g., 'funnel_a', 'default_funnel').
        """
        funnel = self.DEFAULT_FUNNEL_MAP.get(intent, self.DEFAULT_FUNNEL)
        logger.debug(f"Routed intent '{intent}' to funnel '{funnel}'")
        return funnel

    def adjust_bid(self, predicted_cvr: float, cpa_budget: float) -> float:
        """
        Adjust bid based on predicted conversion rate and CPA budget.
        
        Uses a tiered multiplier system based on performance ratio
        (predicted CVR vs target CVR).
        
        Args:
            predicted_cvr: Predicted conversion rate (0.0 to 1.0).
            cpa_budget: Target cost per acquisition in dollars.
            
        Returns:
            Adjusted bid amount in dollars, rounded to 2 decimal places.
            
        Raises:
            ValidationError: If cpa_budget is not positive.
            
        Example:
            >>> bid = agent.adjust_bid(predicted_cvr=0.05, cpa_budget=50.0)
            >>> print(f"Recommended bid: ${bid}")
            'Recommended bid: $13.00'
        """
        cfg = self._config
        
        if cpa_budget <= 0:
            raise ValidationError(
                "CPA budget must be positive",
                {"cpa_budget": cpa_budget}
            )
        
        # Handle zero/negative CVR
        if predicted_cvr <= 0:
            bid = round(cfg.base_bid * cfg.low_cvr_multiplier, 2)
            logger.debug(f"Zero CVR, using minimum bid: ${bid}")
            return bid

        # Calculate target CVR based on budget
        target_cvr = min(
            cfg.max_target_cvr,
            max(cfg.min_target_cvr, cfg.base_bid / cpa_budget)
        )
        
        performance_ratio = predicted_cvr / target_cvr

        # Select multiplier based on performance tier
        if performance_ratio >= cfg.high_performance_ratio:
            multiplier = cfg.high_performance_multiplier
            tier = "high"
        elif performance_ratio >= 1.0:
            multiplier = cfg.medium_performance_multiplier
            tier = "medium"
        elif performance_ratio <= cfg.low_performance_ratio:
            multiplier = cfg.low_performance_multiplier
            tier = "low"
        else:
            multiplier = cfg.below_target_multiplier
            tier = "below_target"

        bid = round(cfg.base_bid * multiplier, 2)
        
        logger.debug(
            f"Bid adjusted: ${bid} (tier: {tier}, ratio: {performance_ratio:.2f})"
        )
        
        return bid

    def retarget_user(
        self,
        user_id: Optional[str],
        last_action: Optional[str]
    ) -> Optional[str]:
        """
        Generate retargeting URL for user based on their last action.
        
        Args:
            user_id: Unique user identifier (must match USER_ID_PATTERN).
            last_action: Last action taken by user (e.g., 'email_submitted').
            
        Returns:
            Retargeting URL string, or None if:
            - user_id is not provided
            - user_id format is invalid
            - last_action is not a recognized retargeting trigger
            
        Example:
            >>> url = agent.retarget_user("user_123", "form_abandoned")
            >>> print(url)
            'https://ads.example.com/retarget/form?uid=user_123'
        """
        if not user_id:
            return None
        
        if not self._validate_user_id(user_id):
            logger.warning(
                "Invalid user_id format rejected",
                extra={"user_id_length": len(user_id)}
            )
            return None
        
        # URL-encode user ID for safety
        safe_uid = quote(user_id, safe='')
        base_url = self._config.retarget_base_url
        
        # Map actions to retargeting URLs
        action_urls = {
            "email_submitted": f"{base_url}/email?uid={safe_uid}",
            "form_abandoned": f"{base_url}/form?uid={safe_uid}",
            "pricing_viewed": f"{base_url}/pricing?uid={safe_uid}",
            "demo_requested": f"{base_url}/demo?uid={safe_uid}",
            "trial_started": f"{base_url}/trial?uid={safe_uid}",
        }
        
        url = action_urls.get(last_action)
        
        if url:
            logger.debug(f"Generated retargeting URL for action '{last_action}'")
        
        return url

    # ------------------------------------------------------------------
    # Data Management
    # ------------------------------------------------------------------
    
    def _load_performance_data(
        self,
        limit: Optional[int] = None
    ) -> List[PerformanceRecord]:
        """
        Load performance data from CSV file.
        
        Thread-safe with configurable record limit to prevent OOM.
        
        Args:
            limit: Maximum number of records to load.
                   Defaults to MAX_CSV_RECORDS.
                   
        Returns:
            List of normalized PerformanceRecord dictionaries.
        """
        if limit is None:
            limit = MAX_CSV_RECORDS
        
        if not self._data_path.exists():
            logger.debug(f"Performance data file not found: {self._data_path}")
            return []
        
        try:
            with self._data_lock:
                with self._data_path.open("r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    records: List[PerformanceRecord] = []
                    
                    for i, row in enumerate(reader):
                        if i >= limit:
                            logger.warning(
                                f"Data load truncated at {limit} records"
                            )
                            break
                        records.append(_normalize_record(row))
            
            logger.debug(f"Loaded {len(records)} performance records")
            return records
            
        except csv.Error as e:
            logger.error(f"CSV parsing error: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            return []

    def _write_performance_data(
        self,
        records: Iterable[PerformanceRecord]
    ) -> None:
        """
        Atomically write performance data to CSV file.
        
        Uses a write-to-temp-then-rename pattern to prevent
        data corruption. Creates a backup of existing data.
        
        Thread-safe.
        
        Args:
            records: Iterable of PerformanceRecord dictionaries.
            
        Raises:
            IOError: If write operation fails.
        """
        rows = list(records)
        if not rows:
            logger.debug("No records to write")
            return
        
        with self._data_lock:
            temp_path = self._data_path.with_suffix('.tmp')
            backup_path = self._data_path.with_suffix('.bak')
            
            try:
                # Ensure parent directory exists
                self._data_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to temporary file
                with temp_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_FIELD_ORDER)
                    writer.writeheader()
                    
                    for row in rows:
                        writer.writerow({
                            "query_text": row.get("query_text", ""),
                            "intent": row.get("intent", ""),
                            "converted": int(row.get("converted", 0)),
                            "clicks": row.get("clicks", 0),
                            "impressions": row.get("impressions", 0),
                            "cost": row.get("cost", 0),
                        })
                
                # Backup existing file if present
                if self._data_path.exists():
                    import shutil
                    shutil.copy2(str(self._data_path), str(backup_path))
                    logger.debug(f"Created backup at {backup_path}")
                
                # Atomic rename
                temp_path.rename(self._data_path)
                logger.info(f"Wrote {len(rows)} records to {self._data_path}")
                
            except Exception as e:
                # Cleanup temp file on failure
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                
                logger.error(f"Failed to write performance data: {e}")
                raise

    def _estimate_intent_cvr(self, intent: str) -> float:
        """
        Estimate conversion rate for a given intent based on historical data.
        
        Falls back to overall CVR if no intent-specific data exists,
        and to default CVR if no historical data exists at all.
        
        Args:
            intent: Intent category to estimate CVR for.
            
        Returns:
            Estimated conversion rate as a float (0.0 to 1.0).
        """
        records = self._load_performance_data()
        
        if not records:
            logger.debug(f"No historical data, using default CVR: {self._config.default_cvr}")
            return self._config.default_cvr

        # Filter to intent-specific records
        intent_records = [r for r in records if r.get("intent") == intent]
        
        if not intent_records:
            logger.debug(f"No data for intent '{intent}', using overall CVR")
            intent_records = records

        # Calculate CVR
        clicks = sum(float(r.get("clicks", 0)) for r in intent_records)
        conversions = sum(int(r.get("converted", 0)) for r in intent_records)
        
        # Use record count as fallback if no click data
        if clicks <= 0:
            clicks = float(len(intent_records))
        
        cvr = conversions / clicks if clicks > 0 else self._config.default_cvr
        
        logger.debug(
            f"Estimated CVR for '{intent}': {cvr:.2%} "
            f"({conversions} conversions / {clicks:.0f} clicks)"
        )
        
        return cvr

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    
    def train_from_feedback(
        self,
        conversion_data: List[Dict[str, str]]
    ) -> Dict[str, int]:
        """
        Retrain model with new conversion feedback data.
        
        Appends new data to existing dataset and retrains the
        intent classification model.
        
        Args:
            conversion_data: List of conversion records. Each record
                must contain: query_text, intent, converted.
                Optional: clicks, impressions, cost.
                
        Returns:
            Dictionary with training results:
            - records_trained: Total number of records used for training
            
        Raises:
            ValidationError: If conversion_data is empty or malformed.
            
        Example:
            >>> feedback = [
            ...     {"query_text": "automate sales", "intent": "workflow_automation", "converted": "1"},
            ...     {"query_text": "manage projects", "intent": "project_management", "converted": "0"},
            ... ]
            >>> result = agent.train_from_feedback(feedback)
            >>> print(result)
            {'records_trained': 152}
        """
        if not conversion_data:
            raise ValidationError(
                "conversion_data must contain at least one record"
            )

        # Validate and normalize records
        normalized_rows: List[PerformanceRecord] = []
        
        for idx, row in enumerate(conversion_data):
            missing = REQUIRED_DATA_COLUMNS - set(row.keys())
            if missing:
                raise ValidationError(
                    f"Record {idx} missing required fields: {sorted(missing)}",
                    {"record_index": idx, "missing_fields": sorted(missing)}
                )
            normalized_rows.append(_normalize_record(row))

        # Combine with existing data
        existing = self._load_performance_data()
        combined = existing + normalized_rows
        
        # Persist combined dataset
        self._write_performance_data(combined)

        # Retrain model
        logger.info(f"Training model with {len(combined)} total records")
        
        train_intent_classifier(
            records=[
                {key: str(value) for key, value in record.items()}
                for record in combined
            ],
            model_path=self._model_path,
        )

        # Invalidate cache so next call uses new model
        self.invalidate_model_cache()
        
        logger.info(
            f"Model training complete",
            extra={"records_trained": len(combined), "new_records": len(normalized_rows)}
        )

        return {"records_trained": len(combined)}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    
    def _calculate_reporting_metrics(self) -> MetricsResult:
        """
        Calculate aggregate performance metrics from historical data.
        
        Returns:
            MetricsResult dictionary with:
            - CTR: Click-through rate (clicks / impressions)
            - CVR: Conversion rate (conversions / clicks)
            - CPA: Cost per acquisition (cost / conversions)
            - Leads: Total conversion count
        """
        records = self._load_performance_data()
        
        if not records:
            return MetricsResult(CTR=0.0, CVR=0.0, CPA=0.0, Leads=0)

        # Aggregate metrics
        impressions = sum(float(r.get("impressions", 0)) for r in records)
        clicks = sum(float(r.get("clicks", 0)) for r in records)
        conversions = sum(int(r.get("converted", 0)) for r in records)
        cost = sum(float(r.get("cost", 0)) for r in records)

        # Fallbacks for missing data
        if impressions <= 0:
            impressions = max(clicks, float(len(records)))
        if clicks <= 0:
            clicks = float(len(records))

        # Calculate rates
        ctr = clicks / impressions if impressions > 0 else 0.0
        cvr = conversions / clicks if clicks > 0 else 0.0
        cpa = cost / conversions if conversions > 0 else 0.0

        return MetricsResult(
            CTR=round(ctr, 4),
            CVR=round(cvr, 4),
            CPA=round(cpa, 2),
            Leads=int(conversions),
        )

    def report_performance(self) -> Dict[str, str]:
        """
        Calculate and report performance metrics.
        
        Optionally updates an external dashboard if configured.
        
        Returns:
            Formatted metrics dictionary with string values
            suitable for display.
        """
        metrics = self._calculate_reporting_metrics()
        formatted = format_metrics(dict(metrics))
        
        # Update dashboard if available
        if DASHBOARD_AVAILABLE and update_dashboard is not None:
            credentials_path = getattr(CONFIG, 'GOOGLE_CREDENTIALS_PATH', None)
            creds = Path(credentials_path) if credentials_path else None
            
            try:
                update_dashboard(formatted, credentials_path=creds)
                logger.info("Dashboard updated successfully")
            except Exception as e:
                logger.error(f"Failed to update dashboard: {e}")
        else:
            logger.debug("Dashboard update skipped - not configured")
        
        return formatted

    # ------------------------------------------------------------------
    # Public Entry Points
    # ------------------------------------------------------------------
    
    def run(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        last_action: Optional[str] = None,
        cpa_budget: Optional[float] = None,
    ) -> PipelineResult:
        """
        Execute the complete SAAStelligence pipeline (synchronous).
        
        This is the main entry point for the agent. It orchestrates
        all pipeline stages and returns a comprehensive result.
        
        For async contexts (e.g., FastAPI), use run_async() instead.
        
        Args:
            user_query: User's search query or input text.
            user_id: Optional user identifier for retargeting.
            last_action: Optional last action for retargeting logic.
            cpa_budget: Optional CPA budget (defaults to config value).
            
        Returns:
            PipelineResult dictionary containing:
            - intent: Detected intent category
            - intent_confidence: Confidence score (0-1)
            - ad_copy: Generated ad text
            - funnel: Target conversion funnel
            - predicted_cvr: Estimated conversion rate
            - bid: Recommended bid amount
            - retarget_url: Retargeting URL (if applicable)
            
        Raises:
            ValidationError: If inputs are invalid.
            ModelNotFoundError: If model is unavailable.
            
        Example:
            >>> result = agent.run(
            ...     user_query="I need to automate my sales pipeline",
            ...     user_id="user_abc123",
            ...     last_action="pricing_viewed"
            ... )
            >>> print(f"Intent: {result['intent']}")
            >>> print(f"Bid: ${result['bid']}")
        """
        # Use default CPA budget if not provided
        if cpa_budget is None:
            cpa_budget = self._config.default_cpa_budget
        
        # Execute pipeline stages
        intent, confidence = self.detect_intent(user_query)
        ad_copy = self.generate_ad(intent)
        funnel = self.route_to_funnel(intent)
        predicted_cvr = self._estimate_intent_cvr(intent)
        bid = self.adjust_bid(predicted_cvr=predicted_cvr, cpa_budget=cpa_budget)
        retarget_url = self.retarget_user(user_id, last_action)
        
        # Build result
        result = PipelineResult(
            intent=intent,
            intent_confidence=round(confidence, 4),
            ad_copy=ad_copy,
            funnel=funnel,
            predicted_cvr=round(predicted_cvr, 4),
            bid=bid,
            retarget_url=retarget_url,
        )
        
        logger.info(
            f"Pipeline complete",
            extra={
                "intent": intent,
                "confidence": f"{confidence:.1%}",
                "bid": f"${bid}",
                "funnel": funnel,
            }
        )
        
        return result

    async def run_async(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        last_action: Optional[str] = None,
        cpa_budget: Optional[float] = None,
    ) -> PipelineResult:
        """
        Execute the complete SAAStelligence pipeline (async-safe).
        
        This method runs the blocking pipeline operations in a thread pool
        to avoid blocking the async event loop. Use this in async contexts
        like FastAPI endpoints.
        
        Args:
            Same as run().
            
        Returns:
            Same as run().
            
        Example:
            >>> result = await agent.run_async(
            ...     user_query="help me manage customer support tickets"
            ... )
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.run(user_query, user_id, last_action, cpa_budget)
        )

    async def report_performance_async(self) -> Dict[str, str]:
        """
        Async-safe version of report_performance.
        
        Runs the blocking report generation in a thread pool.
        
        Returns:
            Same as report_performance().
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.report_performance
        )

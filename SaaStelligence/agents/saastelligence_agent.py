""" Core agent orchestration for the SAAStelligence pipeline."""

from __future__ import annotations

import csv
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote
import re

try:
    from langchain import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LLMChain = None
    OpenAI = None
    PromptTemplate = None
    LANGCHAIN_AVAILABLE = False

from SaaStelligence.config.config import CONFIG
from SaaStelligence.models.train_intent_model import (
    INTENT_CATEGORIES,
    REQUIRED_DATA_COLUMNS,
    load_model as load_intent_model,
    train_intent_classifier,
)
from SaaStelligence.utils.report_utils import format_metrics
from SaaStelligence.utils.reporting_dashboard import update_dashboard  # Fixed typo

logger = logging.getLogger(__name__)

NUM_INTENTS = len(INTENT_CATEGORIES)
OPTIONAL_PERFORMANCE_COLUMNS = ["clicks", "impressions", "cost"]
CSV_FIELD_ORDER = ["query_text", "intent", "converted", *OPTIONAL_PERFORMANCE_COLUMNS]


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class ModelNotFoundError(AgentError):
    """Raised when intent model cannot be loaded or trained."""
    pass


class ValidationError(AgentError):
    """Raised when input validation fails."""
    pass


def _normalize_record(record: Dict[str, str]) -> Dict[str, object]:
    """Normalize a raw CSV record to consistent types."""
    normalized: Dict[str, object] = {
        "query_text": str(record.get("query_text", "")).strip(),
        "intent": str(record.get("intent", "")).strip(),
        "converted": int(float(record.get("converted", 0)) > 0),
    }
    for column in OPTIONAL_PERFORMANCE_COLUMNS:
        value = record.get(column)
        if value in (None, ""):
            normalized[column] = 0.0
        else:
            try:
                normalized[column] = float(value)
            except (ValueError, TypeError):
                normalized[column] = 0.0
    return normalized


class SAAStelligenceAgent:
    """
    Encapsulates intent detection, ad generation, bidding and reporting.
    
    This agent provides a complete pipeline for:
    - Intent classification from user queries
    - Dynamic ad copy generation (requires OpenAI API key)
    - Funnel routing based on intent
    - Bid adjustment based on predicted CVR
    - Retargeting URL generation
    - Performance reporting and dashboard updates
    
    Example:
        >>> agent = SAAStelligenceAgent()
        >>> result = agent.run("I need help automating my sales workflow")
        >>> print(result['intent'])
        'workflow_automation'
    """
    
    # Bidding Configuration
    BASE_BID: float = 10.0
    DEFAULT_CPA_BUDGET: float = 45.0
    LOW_CVR_MULTIPLIER: float = 0.6
    HIGH_PERFORMANCE_MULTIPLIER: float = 1.3
    MEDIUM_PERFORMANCE_MULTIPLIER: float = 1.15
    LOW_PERFORMANCE_MULTIPLIER: float = 0.7
    BELOW_TARGET_MULTIPLIER: float = 0.9
    
    # CVR Thresholds
    MIN_TARGET_CVR: float = 0.02
    MAX_TARGET_CVR: float = 0.15
    DEFAULT_CVR: float = 0.03
    
    # Performance Thresholds
    HIGH_PERFORMANCE_RATIO: float = 1.2
    LOW_PERFORMANCE_RATIO: float = 0.6
    
    # Validation
    MAX_QUERY_LENGTH: int = 1000
    USER_ID_PATTERN: re.Pattern = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
    
    # URLs
    RETARGET_BASE_URL: str = "https://ads.example.com/retarget"
    
    # Default Funnel Mapping
    DEFAULT_FUNNEL_MAP: Dict[str, str] = {
        "workflow_automation": "funnel_a",
        "sales_team_efficiency": "funnel_b",
        "project_management": "funnel_c",
        "customer_support": "funnel_d",
        "marketing_automation": "funnel_e",
    }
    DEFAULT_FUNNEL: str = "default_funnel"

    def __init__(self) -> None:
        """Initialize the SAAStelligence agent."""
        self.intent_mapping = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
        self._intent_model = None
        self._model_path = Path(CONFIG.INTENT_MODEL_PATH)
        self._ad_chain = self._build_ad_chain()
        self._data_path = Path(CONFIG.CONVERSIONS_DATA_PATH)
        
        logger.info("SAAStelligenceAgent initialized")

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _build_ad_chain(self) -> Optional["LLMChain"]:
        """Build LangChain ad generation chain if dependencies available."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, ad generation disabled")
            return None
            
        if not CONFIG.OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured, ad generation disabled")
            return None

        ad_prompt_template = PromptTemplate.from_template(
            """Based on this intent: {intent}, generate a high-conversion ad copy for SaaS lead gen.
            Make it emotionally engaging, include urgency or scarcity where appropriate.
            Output only the ad text."""
        )
        
        logger.info("Ad generation chain configured")
        return LLMChain(
            llm=OpenAI(openai_api_key=CONFIG.OPENAI_API_KEY, temperature=0.7),
            prompt=ad_prompt_template,
        )

    def _ensure_model(self):
        """Ensure intent model is loaded, training if necessary."""
        if self._intent_model is not None:
            return self._intent_model
            
        if self._model_path.exists():
            logger.info(f"Loading intent model from {self._model_path}")
            self._intent_model = load_intent_model(self._model_path)
            return self._intent_model
        
        logger.warning(f"Model not found at {self._model_path}, attempting to train")
        dataset = self._load_performance_data()
        
        if not dataset:
            raise ModelNotFoundError(
                f"Intent model not found at {self._model_path} and no dataset available "
                f"to train a fallback model. Populate {self._data_path} and run the training script."
            )
        
        self._intent_model = train_intent_classifier(
            records=[{key: str(value) for key, value in record.items()} for record in dataset],
            model_path=self._model_path,
        )
        logger.info("Intent model trained successfully")
        return self._intent_model

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def _validate_query(self, query: str) -> str:
        """Validate and sanitize query input."""
        if not isinstance(query, str):
            raise ValidationError("Query must be a string")
        
        query = query.strip()
        if not query:
            raise ValidationError("Query must be a non-empty string")
        
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValidationError(f"Query exceeds maximum length of {self.MAX_QUERY_LENGTH}")
        
        return query

    def _validate_user_id(self, user_id: str) -> bool:
        """Validate user ID format for security."""
        return bool(self.USER_ID_PATTERN.match(user_id))

    # ------------------------------------------------------------------
    # Core agent capabilities
    # ------------------------------------------------------------------
    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect user intent from query text.
        
        Args:
            query: User's search query or input text.
            
        Returns:
            Tuple of (intent_label, confidence_score).
            
        Raises:
            ValidationError: If query is invalid.
            ModelNotFoundError: If model cannot be loaded.
        """
        query = self._validate_query(query)
        logger.debug(f"Detecting intent for query: {query[:50]}...")
        
        model = self._ensure_model()
        probabilities = model.predict_proba(query)
        intent_index = max(range(NUM_INTENTS), key=lambda idx: probabilities[idx])
        intent = INTENT_CATEGORIES[intent_index]
        confidence = float(probabilities[intent_index])
        
        logger.info(f"Detected intent: {intent} (confidence: {confidence:.2%})")
        return intent, confidence

    def generate_ad(self, intent: str) -> str:
        """
        Generate ad copy for the given intent.
        
        Args:
            intent: Detected user intent.
            
        Returns:
            Generated ad copy or fallback message.
        """
        if self._ad_chain is None:
            return (
                "OpenAI API key not configured. "
                "Provide OPENAI_API_KEY to enable dynamic ad copy generation."
            )
        
        try:
            ad_copy = self._ad_chain.run(intent=intent)
            logger.debug(f"Generated ad copy for intent '{intent}'")
            return ad_copy
        except Exception as e:
            logger.error(f"Ad generation failed: {e}")
            return f"Error generating ad copy: {str(e)}"

    def route_to_funnel(self, intent: str) -> str:
        """
        Route user to appropriate conversion funnel based on intent.
        
        Args:
            intent: Detected user intent.
            
        Returns:
            Funnel identifier string.
        """
        funnel = self.DEFAULT_FUNNEL_MAP.get(intent, self.DEFAULT_FUNNEL)
        logger.debug(f"Routed intent '{intent}' to funnel '{funnel}'")
        return funnel

    def adjust_bid(self, predicted_cvr: float, cpa_budget: float) -> float:
        """
        Adjust bid based on predicted conversion rate and CPA budget.
        
        Args:
            predicted_cvr: Predicted conversion rate (0.0 to 1.0).
            cpa_budget: Target cost per acquisition.
            
        Returns:
            Adjusted bid amount.
            
        Raises:
            ValidationError: If cpa_budget is not positive.
        """
        if cpa_budget <= 0:
            raise ValidationError("cpa_budget must be positive")
        
        if predicted_cvr <= 0:
            return round(self.BASE_BID * self.LOW_CVR_MULTIPLIER, 2)

        target_cvr = min(
            self.MAX_TARGET_CVR,
            max(self.MIN_TARGET_CVR, self.BASE_BID / cpa_budget)
        )
        performance_ratio = predicted_cvr / target_cvr

        if performance_ratio >= self.HIGH_PERFORMANCE_RATIO:
            multiplier = self.HIGH_PERFORMANCE_MULTIPLIER
        elif performance_ratio >= 1.0:
            multiplier = self.MEDIUM_PERFORMANCE_MULTIPLIER
        elif performance_ratio <= self.LOW_PERFORMANCE_RATIO:
            multiplier = self.LOW_PERFORMANCE_MULTIPLIER
        else:
            multiplier = self.BELOW_TARGET_MULTIPLIER

        bid = round(self.BASE_BID * multiplier, 2)
        logger.debug(f"Adjusted bid to {bid} (CVR: {predicted_cvr:.2%}, ratio: {performance_ratio:.2f})")
        return bid

    def retarget_user(self, user_id: Optional[str], last_action: Optional[str]) -> Optional[str]:
        """
        Generate retargeting URL for user based on their last action.
        
        Args:
            user_id: Unique user identifier.
            last_action: Last action taken by user.
            
        Returns:
            Retargeting URL or None if not applicable.
        """
        if not user_id:
            return None
            
        if not self._validate_user_id(user_id):
            logger.warning(f"Invalid user_id format rejected: {user_id[:20]}...")
            return None
        
        safe_uid = quote(user_id, safe='')
        
        action_urls = {
            "email_submitted": f"{self.RETARGET_BASE_URL}/email?uid={safe_uid}",
            "form_abandoned": f"{self.RETARGET_BASE_URL}/form?uid={safe_uid}",
        }
        
        return action_urls.get(last_action)

    # ------------------------------------------------------------------
    # Data, training, and reporting helpers
    # ------------------------------------------------------------------
    def _load_performance_data(self) -> List[Dict[str, object]]:
        """Load performance data from CSV file."""
        if not self._data_path.exists():
            logger.debug(f"Performance data file not found: {self._data_path}")
            return []
        
        try:
            with self._data_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                records = [_normalize_record(row) for row in reader]
            logger.debug(f"Loaded {len(records)} performance records")
            return records
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
            return []

    def _write_performance_data(self, records: Iterable[Dict[str, object]]) -> None:
        """Atomically write performance data to prevent data loss."""
        rows = list(records)
        if not rows:
            return
        
        temp_path = self._data_path.with_suffix('.tmp')
        try:
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
            
            # Atomic rename
            shutil.move(str(temp_path), str(self._data_path))
            logger.info(f"Wrote {len(rows)} records to {self._data_path}")
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to write performance data: {e}")
            raise

    def _estimate_intent_cvr(self, intent: str) -> float:
        """Estimate conversion rate for a given intent based on historical data."""
        records = self._load_performance_data()
        if not records:
            return self.DEFAULT_CVR

        intent_records = [r for r in records if r.get("intent") == intent]
        if not intent_records:
            intent_records = records  # Fallback to all records

        clicks = sum(float(r.get("clicks", 0)) for r in intent_records)
        conversions = sum(int(r.get("converted", 0)) for r in intent_records)
        
        # Use record count as fallback if no click data
        if clicks <= 0:
            clicks = float(len(intent_records))
        
        return conversions / clicks if clicks > 0 else self.DEFAULT_CVR

    def train_from_feedback(self, conversion_data: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Retrain model with new conversion feedback data.
        
        Args:
            conversion_data: List of conversion records with required fields.
            
        Returns:
            Dictionary with training results.
            
        Raises:
            ValidationError: If conversion_data is invalid.
        """
        if not conversion_data:
            raise ValidationError("conversion_data must contain at least one record")

        normalized_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(conversion_data):
            missing = REQUIRED_DATA_COLUMNS - set(row.keys())
            if missing:
                raise ValidationError(
                    f"Record {idx} missing required fields: {sorted(missing)}"
                )
            normalized_rows.append(_normalize_record(row))

        existing = self._load_performance_data()
        combined = existing + normalized_rows
        self._write_performance_data(combined)

        logger.info(f"Training model with {len(combined)} total records")
        train_intent_classifier(
            records=[{key: str(value) for key, value in record.items()} for record in combined],
            model_path=self._model_path,
        )

        # Reset cached model so subsequent calls use updated version
        self._intent_model = None
        logger.info("Model training complete, cache invalidated")

        return {"records_trained": len(combined)}

    def _calculate_reporting_metrics(self) -> Dict[str, float]:
        """Calculate aggregate performance metrics."""
        records = self._load_performance_data()
        if not records:
            return {"CTR": 0.0, "CVR": 0.0, "CPA": 0.0, "Leads": 0}

        impressions = sum(float(r.get("impressions", 0)) for r in records)
        clicks = sum(float(r.get("clicks", 0)) for r in records)
        conversions = sum(int(r.get("converted", 0)) for r in records)
        cost = sum(float(r.get("cost", 0)) for r in records)

        # Fallbacks for missing data
        if impressions <= 0:
            impressions = max(clicks, float(len(records)))
        if clicks <= 0:
            clicks = float(len(records))

        ctr = clicks / impressions if impressions > 0 else 0.0
        cvr = conversions / clicks if clicks > 0 else 0.0
        cpa = cost / conversions if conversions > 0 else 0.0

        return {
            "CTR": round(ctr, 4),
            "CVR": round(cvr, 4),
            "CPA": round(cpa, 2),
            "Leads": int(conversions),
        }

    def report_performance(self) -> Dict[str, str]:
        """
        Calculate and report performance metrics to dashboard.
        
        Returns:
            Formatted metrics dictionary.
        """
        metrics = self._calculate_reporting_metrics()
        formatted = format_metrics(metrics)
        
        credentials_path = (
            Path(CONFIG.GOOGLE_CREDENTIALS_PATH)
            if CONFIG.GOOGLE_CREDENTIALS_PATH
            else None
        )
        
        try:
            update_dashboard(formatted, credentials_path=credentials_path)
            logger.info("Dashboard updated successfully")
        except Exception as e:
            logger.error(f"Failed to update dashboard: {e}")
        
        return formatted

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        last_action: Optional[str] = None,
        cpa_budget: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Execute the complete SAAStelligence pipeline.
        
        Args:
            user_query: User's search query or input text.
            user_id: Optional user identifier for retargeting.
            last_action: Optional last action for retargeting logic.
            cpa_budget: Optional CPA budget (defaults to DEFAULT_CPA_BUDGET).
            
        Returns:
            Dictionary with intent, ad copy, funnel, bid, and retargeting info.
        """
        if cpa_budget is None:
            cpa_budget = self.DEFAULT_CPA_BUDGET
            
        intent, confidence = self.detect_intent(user_query)
        ad_copy = self.generate_ad(intent)
        funnel = self.route_to_funnel(intent)
        predicted_cvr = self._estimate_intent_cvr(intent)
        bid = self.adjust_bid(predicted_cvr=predicted_cvr, cpa_budget=cpa_budget)
        retarget_url = self.retarget_user(user_id, last_action)
        
        result = {
            "intent": intent,
            "intent_confidence": round(confidence, 4),
            "ad_copy": ad_copy,
            "funnel": funnel,
            "predicted_cvr": round(predicted_cvr, 4),
            "bid": bid,
            "retarget_url": retarget_url,
        }
        
        logger.info(f"Pipeline complete: intent={intent}, bid={bid}")
        return result

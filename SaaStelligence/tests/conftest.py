"""
Pytest fixtures and configuration for SAAStelligence tests.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "SaaStelligence"))


# ============== MOCK CLASSES ==============

class MockIntentModel:
    """Mock intent classification model for testing."""
    
    def __init__(self, default_intent_idx: int = 0, confidence: float = 0.85):
        self.default_intent_idx = default_intent_idx
        self.confidence = confidence
        self.call_count = 0
    
    def predict_proba(self, query: str) -> List[float]:
        """Return mock probability distribution."""
        self.call_count += 1
        # 5 intent categories
        probs = [0.05, 0.05, 0.05, 0.05, 0.05]
        probs[self.default_intent_idx] = self.confidence
        return probs


class MockLLMChain:
    """Mock LangChain LLM chain for testing."""
    
    def __init__(self, response: str = "Generated ad copy for testing."):
        self.response = response
        self.call_count = 0
        self.last_intent = None
    
    def run(self, intent: str) -> str:
        self.call_count += 1
        self.last_intent = intent
        return self.response


# ============== FIXTURES ==============

@pytest.fixture
def mock_intent_model() -> MockIntentModel:
    """Provide a mock intent model."""
    return MockIntentModel()


@pytest.fixture
def mock_llm_chain() -> MockLLMChain:
    """Provide a mock LLM chain."""
    return MockLLMChain()


@pytest.fixture
def sample_performance_records() -> List[Dict[str, Any]]:
    """Provide sample performance data records."""
    return [
        {
            "query_text": "automate my sales workflow",
            "intent": "workflow_automation",
            "converted": 1,
            "clicks": 100,
            "impressions": 1000,
            "cost": 50.0,
        },
        {
            "query_text": "improve sales team efficiency",
            "intent": "sales_team_efficiency",
            "converted": 0,
            "clicks": 80,
            "impressions": 800,
            "cost": 40.0,
        },
        {
            "query_text": "project management tools",
            "intent": "project_management",
            "converted": 1,
            "clicks": 120,
            "impressions": 1200,
            "cost": 60.0,
        },
        {
            "query_text": "customer support automation",
            "intent": "customer_support",
            "converted": 1,
            "clicks": 90,
            "impressions": 900,
            "cost": 45.0,
        },
        {
            "query_text": "marketing automation platform",
            "intent": "marketing_automation",
            "converted": 0,
            "clicks": 70,
            "impressions": 700,
            "cost": 35.0,
        },
    ]


@pytest.fixture
def sample_training_data() -> List[Dict[str, str]]:
    """Provide sample training data in string format."""
    return [
        {
            "query_text": "automate sales pipeline",
            "intent": "workflow_automation",
            "converted": "1",
        },
        {
            "query_text": "manage customer tickets",
            "intent": "customer_support",
            "converted": "1",
        },
        {
            "query_text": "email marketing automation",
            "intent": "marketing_automation",
            "converted": "0",
        },
    ]


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def mock_config(temp_data_dir: Path):
    """Provide a mock CONFIG object."""
    config = MagicMock()
    config.INTENT_MODEL_PATH = str(temp_data_dir / "intent_model.pkl")
    config.CONVERSIONS_DATA_PATH = str(temp_data_dir / "conversions.csv")
    config.BASE_BID = 10.0
    config.DEFAULT_CPA_BUDGET = 45.0
    config.RETARGET_BASE_URL = "https://test.example.com/retarget"
    config.MAX_QUERY_LENGTH = 1000
    config.MIN_CONFIDENCE_THRESHOLD = 0.3
    config.MODEL_CACHE_TTL = 3600
    config.AGENT_MAX_WORKERS = 2
    config.OPENAI_API_KEY = None  # Disabled by default
    config.GOOGLE_CREDENTIALS_PATH = None
    return config


@pytest.fixture
def agent_with_mocks(
    mock_config,
    mock_intent_model: MockIntentModel,
) -> Generator:
    """
    Create a SAAStelligenceAgent with mocked dependencies.
    
    Yields the agent and mock objects for assertion.
    """
    with patch("agents.saastelligence_agent.CONFIG", mock_config):
        with patch(
            "agents.saastelligence_agent.load_intent_model",
            return_value=mock_intent_model,
        ):
            # Create a mock model file so _ensure_model doesn't try to train
            model_path = Path(mock_config.INTENT_MODEL_PATH)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.touch()
            
            from agents.saastelligence_agent import SAAStelligenceAgent
            
            agent = SAAStelligenceAgent()
            agent._ad_chain = None  # Disable ad generation for unit tests
            
            yield {
                "agent": agent,
                "model": mock_intent_model,
                "config": mock_config,
            }


@pytest.fixture
def agent_with_ad_generation(
    mock_config,
    mock_intent_model: MockIntentModel,
    mock_llm_chain: MockLLMChain,
) -> Generator:
    """Create agent with mocked ad generation."""
    with patch("agents.saastelligence_agent.CONFIG", mock_config):
        with patch(
            "agents.saastelligence_agent.load_intent_model",
            return_value=mock_intent_model,
        ):
            model_path = Path(mock_config.INTENT_MODEL_PATH)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.touch()
            
            from agents.saastelligence_agent import SAAStelligenceAgent
            
            agent = SAAStelligenceAgent()
            agent._ad_chain = mock_llm_chain
            
            yield {
                "agent": agent,
                "model": mock_intent_model,
                "chain": mock_llm_chain,
            }


# ============== PYTEST CONFIGURATION ==============

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API keys"
    )
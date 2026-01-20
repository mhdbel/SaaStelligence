"""
Unit tests for helper functions and utilities.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "SaaStelligence"))


class TestSanitizeString:
    """Tests for _sanitize_string helper."""
    
    def test_sanitize_normal_string(self):
        from agents.saastelligence_agent import _sanitize_string
        
        result = _sanitize_string("Hello World")
        assert result == "Hello World"
    
    def test_sanitize_strips_whitespace(self):
        from agents.saastelligence_agent import _sanitize_string
        
        result = _sanitize_string("  Hello World  ")
        assert result == "Hello World"
    
    @pytest.mark.parametrize("dangerous_char", ["=", "+", "-", "@", "\t", "\r", "\n", "|"])
    def test_sanitize_removes_dangerous_prefix(self, dangerous_char):
        from agents.saastelligence_agent import _sanitize_string
        
        result = _sanitize_string(f"{dangerous_char}DANGEROUS")
        assert not result.startswith(dangerous_char)
    
    def test_sanitize_removes_multiple_dangerous_chars(self):
        from agents.saastelligence_agent import _sanitize_string
        
        result = _sanitize_string("=+@-DANGEROUS")
        assert result == "DANGEROUS"
    
    def test_sanitize_truncates_long_string(self):
        from agents.saastelligence_agent import _sanitize_string
        
        long_string = "x" * 2000
        result = _sanitize_string(long_string, max_length=100)
        assert len(result) == 100
    
    def test_sanitize_empty_string(self):
        from agents.saastelligence_agent import _sanitize_string
        
        assert _sanitize_string("") == ""
        assert _sanitize_string(None) == ""


class TestSafeFloat:
    """Tests for _safe_float helper."""
    
    def test_safe_float_valid_number(self):
        from agents.saastelligence_agent import _safe_float
        
        assert _safe_float("123.45") == 123.45
        assert _safe_float(123.45) == 123.45
        assert _safe_float(100) == 100.0
    
    def test_safe_float_invalid_returns_default(self):
        from agents.saastelligence_agent import _safe_float
        
        assert _safe_float("not a number") == 0.0
        assert _safe_float("abc", default=5.0) == 5.0
    
    def test_safe_float_none_returns_default(self):
        from agents.saastelligence_agent import _safe_float
        
        assert _safe_float(None) == 0.0
        assert _safe_float(None, default=10.0) == 10.0
    
    def test_safe_float_empty_string_returns_default(self):
        from agents.saastelligence_agent import _safe_float
        
        assert _safe_float("") == 0.0
    
    def test_safe_float_guards_against_infinity(self):
        from agents.saastelligence_agent import _safe_float
        
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0


class TestHashQuery:
    """Tests for _hash_query helper."""
    
    def test_hash_query_returns_string(self):
        from agents.saastelligence_agent import _hash_query
        
        result = _hash_query("test query")
        assert isinstance(result, str)
    
    def test_hash_query_consistent(self):
        from agents.saastelligence_agent import _hash_query
        
        result1 = _hash_query("test query")
        result2 = _hash_query("test query")
        assert result1 == result2
    
    def test_hash_query_different_for_different_inputs(self):
        from agents.saastelligence_agent import _hash_query
        
        result1 = _hash_query("query one")
        result2 = _hash_query("query two")
        assert result1 != result2
    
    def test_hash_query_truncated(self):
        from agents.saastelligence_agent import _hash_query
        
        result = _hash_query("test")
        assert len(result) == 8  # Truncated hash


class TestNormalizeRecord:
    """Tests for _normalize_record helper."""
    
    def test_normalize_complete_record(self):
        from agents.saastelligence_agent import _normalize_record
        
        record = {
            "query_text": "test query",
            "intent": "workflow_automation",
            "converted": "1",
            "clicks": "100",
            "impressions": "1000",
            "cost": "50.5",
        }
        
        result = _normalize_record(record)
        
        assert result["query_text"] == "test query"
        assert result["intent"] == "workflow_automation"
        assert result["converted"] == 1
        assert result["clicks"] == 100.0
        assert result["impressions"] == 1000.0
        assert result["cost"] == 50.5
    
    def test_normalize_missing_optional_fields(self):
        from agents.saastelligence_agent import _normalize_record
        
        record = {
            "query_text": "test",
            "intent": "test_intent",
            "converted": "0",
        }
        
        result = _normalize_record(record)
        
        assert result["clicks"] == 0.0
        assert result["impressions"] == 0.0
        assert result["cost"] == 0.0
    
    def test_normalize_sanitizes_dangerous_input(self):
        from agents.saastelligence_agent import _normalize_record
        
        record = {
            "query_text": "=DANGEROUS FORMULA",
            "intent": "+alert(1)",
            "converted": "1",
        }
        
        result = _normalize_record(record)
        
        assert not result["query_text"].startswith("=")
        assert not result["intent"].startswith("+")


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""
    
    def test_config_defaults(self):
        from agents.saastelligence_agent import AgentConfig
        
        config = AgentConfig()
        
        assert config.base_bid == 10.0
        assert config.default_cpa_budget == 45.0
        assert config.max_query_length == 1000
    
    def test_config_from_config_with_values(self):
        from agents.saastelligence_agent import AgentConfig
        from unittest.mock import MagicMock
        
        mock_config = MagicMock()
        mock_config.BASE_BID = 15.0
        mock_config.DEFAULT_CPA_BUDGET = 60.0
        
        config = AgentConfig.from_config(mock_config)
        
        assert config.base_bid == 15.0
        assert config.default_cpa_budget == 60.0
    
    def test_config_from_config_missing_values(self):
        from agents.saastelligence_agent import AgentConfig
        from unittest.mock import MagicMock
        
        mock_config = MagicMock(spec=[])  # Empty spec = no attributes
        
        config = AgentConfig.from_config(mock_config)
        
        # Should use defaults
        assert config.base_bid == 10.0
    
    def test_config_is_immutable(self):
        from agents.saastelligence_agent import AgentConfig
        
        config = AgentConfig()
        
        with pytest.raises(AttributeError):
            config.base_bid = 20.0
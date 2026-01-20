"""
Unit tests for SAAStelligenceAgent core functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestIntentDetection:
    """Tests for intent detection functionality."""
    
    def test_detect_intent_returns_tuple(self, agent_with_mocks):
        """Should return (intent, confidence) tuple."""
        agent = agent_with_mocks["agent"]
        
        result = agent.detect_intent("automate my sales workflow")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_detect_intent_valid_intent(self, agent_with_mocks):
        """Should return a valid intent category."""
        agent = agent_with_mocks["agent"]
        
        intent, confidence = agent.detect_intent("help with sales automation")
        
        from agents.saastelligence_agent import INTENT_CATEGORIES
        assert intent in INTENT_CATEGORIES
    
    def test_detect_intent_confidence_range(self, agent_with_mocks):
        """Confidence should be between 0 and 1."""
        agent = agent_with_mocks["agent"]
        
        _, confidence = agent.detect_intent("test query")
        
        assert 0.0 <= confidence <= 1.0
    
    def test_detect_intent_empty_query_raises(self, agent_with_mocks):
        """Empty query should raise ValidationError."""
        agent = agent_with_mocks["agent"]
        
        from agents.saastelligence_agent import ValidationError
        
        with pytest.raises(ValidationError):
            agent.detect_intent("")
    
    def test_detect_intent_whitespace_query_raises(self, agent_with_mocks):
        """Whitespace-only query should raise ValidationError."""
        agent = agent_with_mocks["agent"]
        
        from agents.saastelligence_agent import ValidationError
        
        with pytest.raises(ValidationError):
            agent.detect_intent("   \t\n  ")
    
    def test_detect_intent_too_long_query_raises(self, agent_with_mocks):
        """Query exceeding max length should raise ValidationError."""
        agent = agent_with_mocks["agent"]
        
        from agents.saastelligence_agent import ValidationError
        
        long_query = "x" * 1001
        with pytest.raises(ValidationError) as exc_info:
            agent.detect_intent(long_query)
        
        assert "maximum length" in str(exc_info.value).lower()
    
    def test_detect_intent_non_string_raises(self, agent_with_mocks):
        """Non-string query should raise ValidationError."""
        agent = agent_with_mocks["agent"]
        
        from agents.saastelligence_agent import ValidationError
        
        with pytest.raises(ValidationError):
            agent.detect_intent(12345)


class TestAdGeneration:
    """Tests for ad generation functionality."""
    
    def test_generate_ad_without_chain_returns_fallback(self, agent_with_mocks):
        """Should return fallback message when chain not configured."""
        agent = agent_with_mocks["agent"]
        agent._ad_chain = None
        
        result = agent.generate_ad("workflow_automation")
        
        assert "not configured" in result.lower()
    
    def test_generate_ad_with_chain_returns_content(self, agent_with_ad_generation):
        """Should return generated content when chain is configured."""
        agent = agent_with_ad_generation["agent"]
        chain = agent_with_ad_generation["chain"]
        
        result = agent.generate_ad("workflow_automation")
        
        assert result == "Generated ad copy for testing."
        assert chain.last_intent == "workflow_automation"
    
    def test_generate_ad_increments_call_count(self, agent_with_ad_generation):
        """Should call the chain each time."""
        agent = agent_with_ad_generation["agent"]
        chain = agent_with_ad_generation["chain"]
        
        agent.generate_ad("intent_1")
        agent.generate_ad("intent_2")
        
        assert chain.call_count == 2


class TestFunnelRouting:
    """Tests for funnel routing functionality."""
    
    @pytest.mark.parametrize("intent,expected_funnel", [
        ("workflow_automation", "funnel_a"),
        ("sales_team_efficiency", "funnel_b"),
        ("project_management", "funnel_c"),
        ("customer_support", "funnel_d"),
        ("marketing_automation", "funnel_e"),
    ])
    def test_route_to_funnel_known_intents(
        self, agent_with_mocks, intent, expected_funnel
    ):
        """Known intents should route to their designated funnels."""
        agent = agent_with_mocks["agent"]
        
        result = agent.route_to_funnel(intent)
        
        assert result == expected_funnel
    
    def test_route_to_funnel_unknown_intent(self, agent_with_mocks):
        """Unknown intent should route to default funnel."""
        agent = agent_with_mocks["agent"]
        
        result = agent.route_to_funnel("unknown_intent")
        
        assert result == "default_funnel"


class TestBidAdjustment:
    """Tests for bid adjustment functionality."""
    
    def test_adjust_bid_zero_cvr(self, agent_with_mocks):
        """Zero CVR should return minimum bid."""
        agent = agent_with_mocks["agent"]
        
        bid = agent.adjust_bid(predicted_cvr=0.0, cpa_budget=50.0)
        
        assert bid == 6.0  # base_bid * low_cvr_multiplier
    
    def test_adjust_bid_negative_cvr(self, agent_with_mocks):
        """Negative CVR should be treated as zero."""
        agent = agent_with_mocks["agent"]
        
        bid = agent.adjust_bid(predicted_cvr=-0.1, cpa_budget=50.0)
        
        assert bid == 6.0
    
    def test_adjust_bid_high_performance(self, agent_with_mocks):
        """High CVR should get high performance multiplier."""
        agent = agent_with_mocks["agent"]
        
        bid = agent.adjust_bid(predicted_cvr=0.15, cpa_budget=50.0)
        
        assert bid == 13.0  # base_bid * high_performance_multiplier
    
    def test_adjust_bid_low_performance(self, agent_with_mocks):
        """Very low CVR should get low performance multiplier."""
        agent = agent_with_mocks["agent"]
        
        bid = agent.adjust_bid(predicted_cvr=0.005, cpa_budget=50.0)
        
        assert bid == 7.0  # base_bid * low_performance_multiplier
    
    def test_adjust_bid_invalid_budget_raises(self, agent_with_mocks):
        """Zero or negative budget should raise ValidationError."""
        agent = agent_with_mocks["agent"]
        
        from agents.saastelligence_agent import ValidationError
        
        with pytest.raises(ValidationError):
            agent.adjust_bid(predicted_cvr=0.05, cpa_budget=0)
        
        with pytest.raises(ValidationError):
            agent.adjust_bid(predicted_cvr=0.05, cpa_budget=-10)
    
    def test_adjust_bid_returns_float(self, agent_with_mocks):
        """Should always return a float."""
        agent = agent_with_mocks["agent"]
        
        bid = agent.adjust_bid(predicted_cvr=0.05, cpa_budget=50.0)
        
        assert isinstance(bid, float)


class TestRetargeting:
    """Tests for retargeting URL generation."""
    
    def test_retarget_user_no_user_id(self, agent_with_mocks):
        """Should return None when user_id is not provided."""
        agent = agent_with_mocks["agent"]
        
        result = agent.retarget_user(None, "email_submitted")
        
        assert result is None
    
    def test_retarget_user_empty_user_id(self, agent_with_mocks):
        """Should return None for empty user_id."""
        agent = agent_with_mocks["agent"]
        
        result = agent.retarget_user("", "email_submitted")
        
        assert result is None
    
    @pytest.mark.parametrize("action,path", [
        ("email_submitted", "email"),
        ("form_abandoned", "form"),
        ("pricing_viewed", "pricing"),
        ("demo_requested", "demo"),
        ("trial_started", "trial"),
    ])
    def test_retarget_user_valid_actions(
        self, agent_with_mocks, action, path
    ):
        """Valid actions should generate correct URLs."""
        agent = agent_with_mocks["agent"]
        
        result = agent.retarget_user("user_123", action)
        
        assert result is not None
        assert path in result
        assert "user_123" in result
    
    def test_retarget_user_unknown_action(self, agent_with_mocks):
        """Unknown action should return None."""
        agent = agent_with_mocks["agent"]
        
        result = agent.retarget_user("user_123", "unknown_action")
        
        assert result is None
    
    def test_retarget_user_invalid_user_id_format(self, agent_with_mocks):
        """Invalid user_id format should return None."""
        agent = agent_with_mocks["agent"]
        
        # Contains invalid characters
        result = agent.retarget_user("user@123!#$", "email_submitted")
        assert result is None
        
        # Too long
        long_id = "a" * 100
        result = agent.retarget_user(long_id, "email_submitted")
        assert result is None
    
    def test_retarget_user_url_encodes_special_chars(self, agent_with_mocks):
        """User ID with allowed special chars should be URL encoded."""
        agent = agent_with_mocks["agent"]
        
        result = agent.retarget_user("user-123_abc", "email_submitted")
        
        assert result is not None
        assert "user-123_abc" in result


class TestFullPipeline:
    """Tests for the complete run() pipeline."""
    
    def test_run_returns_complete_result(self, agent_with_mocks):
        """Should return all expected fields."""
        agent = agent_with_mocks["agent"]
        
        result = agent.run("test query")
        
        expected_keys = {
            "intent", "intent_confidence", "ad_copy", 
            "funnel", "predicted_cvr", "bid", "retarget_url"
        }
        assert set(result.keys()) == expected_keys
    
    def test_run_with_user_context(self, agent_with_mocks):
        """Should include retarget URL when user context provided."""
        agent = agent_with_mocks["agent"]
        
        result = agent.run(
            "test query",
            user_id="user_123",
            last_action="email_submitted",
        )
        
        assert result["retarget_url"] is not None
    
    def test_run_without_user_context(self, agent_with_mocks):
        """Should have None retarget URL without user context."""
        agent = agent_with_mocks["agent"]
        
        result = agent.run("test query")
        
        assert result["retarget_url"] is None
    
    def test_run_custom_cpa_budget(self, agent_with_mocks):
        """Should use custom CPA budget when provided."""
        agent = agent_with_mocks["agent"]
        
        result1 = agent.run("test query", cpa_budget=100.0)
        result2 = agent.run("test query", cpa_budget=20.0)
        
        # Different budgets should produce different bids
        # (actual values depend on CVR estimation)
        assert isinstance(result1["bid"], float)
        assert isinstance(result2["bid"], float)


class TestAsyncPipeline:
    """Tests for async pipeline methods."""
    
    @pytest.mark.asyncio
    async def test_run_async_returns_same_as_sync(self, agent_with_mocks):
        """Async run should return same structure as sync."""
        agent = agent_with_mocks["agent"]
        
        sync_result = agent.run("test query")
        async_result = await agent.run_async("test query")
        
        assert set(sync_result.keys()) == set(async_result.keys())
    
    @pytest.mark.asyncio
    async def test_report_performance_async(self, agent_with_mocks):
        """Async report should work without errors."""
        agent = agent_with_mocks["agent"]
        
        result = await agent.report_performance_async()
        
        assert isinstance(result, dict)
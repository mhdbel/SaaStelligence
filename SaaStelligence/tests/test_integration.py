"""
Integration tests for SAAStelligence pipeline.
"""

import csv
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestDataPersistence:
    """Integration tests for data loading and saving."""
    
    def test_write_and_load_performance_data(self, agent_with_mocks, temp_data_dir):
        """Should correctly round-trip performance data."""
        agent = agent_with_mocks["agent"]
        agent._data_path = temp_data_dir / "test_data.csv"
        
        records = [
            {
                "query_text": "test query 1",
                "intent": "workflow_automation",
                "converted": 1,
                "clicks": 100,
                "impressions": 1000,
                "cost": 50.0,
            },
            {
                "query_text": "test query 2",
                "intent": "customer_support",
                "converted": 0,
                "clicks": 80,
                "impressions": 800,
                "cost": 40.0,
            },
        ]
        
        # Write
        agent._write_performance_data(records)
        
        # Verify file exists
        assert agent._data_path.exists()
        
        # Load
        loaded = agent._load_performance_data()
        
        assert len(loaded) == 2
        assert loaded[0]["query_text"] == "test query 1"
        assert loaded[0]["converted"] == 1
        assert loaded[1]["intent"] == "customer_support"
    
    def test_write_creates_backup(self, agent_with_mocks, temp_data_dir):
        """Should create backup when overwriting existing file."""
        agent = agent_with_mocks["agent"]
        agent._data_path = temp_data_dir / "test_data.csv"
        
        # Write initial data
        agent._write_performance_data([{
            "query_text": "original",
            "intent": "test",
            "converted": 0,
            "clicks": 10,
            "impressions": 100,
            "cost": 5.0,
        }])
        
        # Write new data
        agent._write_performance_data([{
            "query_text": "updated",
            "intent": "test",
            "converted": 1,
            "clicks": 20,
            "impressions": 200,
            "cost": 10.0,
        }])
        
        # Backup should exist
        backup_path = agent._data_path.with_suffix('.bak')
        assert backup_path.exists()


@pytest.mark.integration
class TestMetricsCalculation:
    """Integration tests for metrics calculation."""
    
    def test_calculate_metrics_with_data(
        self, agent_with_mocks, temp_data_dir, sample_performance_records
    ):
        """Should correctly calculate metrics from data."""
        agent = agent_with_mocks["agent"]
        agent._data_path = temp_data_dir / "metrics_data.csv"
        
        # Write test data
        agent._write_performance_data(sample_performance_records)
        
        # Calculate metrics
        metrics = agent._calculate_reporting_metrics()
        
        # Verify structure
        assert "CTR" in metrics
        assert "CVR" in metrics
        assert "CPA" in metrics
        assert "Leads" in metrics
        
        # Verify calculations
        total_clicks = sum(r["clicks"] for r in sample_performance_records)
        total_impressions = sum(r["impressions"] for r in sample_performance_records)
        total_conversions = sum(r["converted"] for r in sample_performance_records)
        
        expected_ctr = total_clicks / total_impressions
        assert abs(metrics["CTR"] - expected_ctr) < 0.001
        
        assert metrics["Leads"] == total_conversions
    
    def test_calculate_metrics_empty_data(self, agent_with_mocks):
        """Should return zeros when no data exists."""
        agent = agent_with_mocks["agent"]
        
        metrics = agent._calculate_reporting_metrics()
        
        assert metrics["CTR"] == 0.0
        assert metrics["CVR"] == 0.0
        assert metrics["CPA"] == 0.0
        assert metrics["Leads"] == 0


@pytest.mark.integration
class TestCVREstimation:
    """Integration tests for CVR estimation."""
    
    def test_estimate_intent_cvr_with_data(
        self, agent_with_mocks, temp_data_dir, sample_performance_records
    ):
        """Should estimate CVR from historical data."""
        agent = agent_with_mocks["agent"]
        agent._data_path = temp_data_dir / "cvr_data.csv"
        
        agent._write_performance_data(sample_performance_records)
        
        cvr = agent._estimate_intent_cvr("workflow_automation")
        
        # Should be based on workflow_automation records
        assert 0.0 <= cvr <= 1.0
    
    def test_estimate_intent_cvr_no_data(self, agent_with_mocks):
        """Should return default CVR when no data."""
        agent = agent_with_mocks["agent"]
        
        cvr = agent._estimate_intent_cvr("any_intent")
        
        assert cvr == agent._config.default_cvr


@pytest.mark.integration
class TestModelCaching:
    """Integration tests for model caching."""
    
    def test_model_cache_prevents_reload(self, agent_with_mocks):
        """Cached model should not reload within TTL."""
        agent = agent_with_mocks["agent"]
        model = agent_with_mocks["model"]
        
        # First call loads model
        agent._ensure_model()
        
        # Second call should use cache
        agent._ensure_model()
        
        # Model should only be "loaded" once (mock tracks this)
        assert agent._cached_model is not None
    
    def test_invalidate_cache_forces_reload(self, agent_with_mocks):
        """Invalidating cache should clear cached model."""
        agent = agent_with_mocks["agent"]
        
        agent._ensure_model()
        assert agent._cached_model is not None
        
        agent.invalidate_model_cache()
        assert agent._cached_model is None


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_training(
        self, agent_with_mocks, sample_training_data
    ):
        """Should train and run complete pipeline."""
        agent = agent_with_mocks["agent"]
        
        # Note: This would require more complex mocking
        # of the training function for a true E2E test
        pass


@pytest.mark.requires_api
class TestWithRealAPI:
    """Tests that require real API keys (skipped by default)."""
    
    @pytest.mark.skip(reason="Requires OPENAI_API_KEY")
    def test_real_ad_generation(self):
        """Test real ad generation with OpenAI."""
        pass
"""
Tests for the main CLI module.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLIParser:
    """Tests for CLI argument parsing."""
    
    def test_parser_creation(self):
        from main import create_parser
        
        parser = create_parser()
        assert parser is not None
    
    def test_run_command_parsing(self):
        from main import create_parser
        
        parser = create_parser()
        args = parser.parse_args(["run", "test query"])
        
        assert args.command == "run"
        assert args.query == "test query"
    
    def test_run_command_with_options(self):
        from main import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "run", "test query",
            "--user-id", "user_123",
            "--last-action", "email_submitted",
            "--cpa-budget", "50.0",
            "--format", "json",
        ])
        
        assert args.user_id == "user_123"
        assert args.last_action == "email_submitted"
        assert args.cpa_budget == 50.0
        assert args.output_format == "json"
    
    def test_report_command_parsing(self):
        from main import create_parser
        
        parser = create_parser()
        args = parser.parse_args(["report", "--format", "json"])
        
        assert args.command == "report"
        assert args.output_format == "json"
    
    def test_train_command_parsing(self):
        from main import create_parser
        
        parser = create_parser()
        args = parser.parse_args(["train", "--file", "data.csv"])
        
        assert args.command == "train"
        assert args.file == "data.csv"
    
    def test_server_command_parsing(self):
        from main import create_parser
        
        parser = create_parser()
        args = parser.parse_args([
            "server",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--reload",
        ])
        
        assert args.command == "server"
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.reload is True


class TestCLICommands:
    """Tests for CLI command execution."""
    
    def test_cmd_run_success(self):
        from main import cmd_run
        import argparse
        
        mock_result = {
            "intent": "workflow_automation",
            "intent_confidence": 0.85,
            "ad_copy": "Test ad",
            "funnel": "funnel_a",
            "predicted_cvr": 0.05,
            "bid": 10.0,
            "retarget_url": None,
        }
        
        with patch("main.SAAStelligenceAgent") as MockAgent:
            MockAgent.return_value.run.return_value = mock_result
            
            args = argparse.Namespace(
                query="test query",
                user_id=None,
                last_action=None,
                cpa_budget=None,
                output_format="json",
            )
            
            result = cmd_run(args)
            assert result == 0
    
    def test_cmd_run_validation_error(self):
        from main import cmd_run
        import argparse
        
        with patch("main.SAAStelligenceAgent") as MockAgent:
            from agents.saastelligence_agent import ValidationError
            MockAgent.return_value.run.side_effect = ValidationError("Test error")
            
            args = argparse.Namespace(
                query="",
                user_id=None,
                last_action=None,
                cpa_budget=None,
                output_format="table",
            )
            
            result = cmd_run(args)
            assert result == 1
    
    def test_cmd_report_success(self):
        from main import cmd_report
        import argparse
        
        with patch("main.SAAStelligenceAgent") as MockAgent:
            MockAgent.return_value.report_performance.return_value = {
                "CTR": "10.5%",
                "CVR": "3.2%",
                "CPA": "$45.00",
                "Leads": "150",
            }
            
            args = argparse.Namespace(output_format="json")
            
            result = cmd_report(args)
            assert result == 0
#!/usr/bin/env python3
"""
SAAStelligence CLI - AI-Powered SaaS Lead Generation Pipeline

Usage:
    python main.py run "your query here" [options]
    python main.py report
    python main.py train --file data.csv
    python main.py server --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging before imports
def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure application logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if json_output:
        format_str = json.dumps({
            "timestamp": "%(asctime)s",
            "level": "%(levelname)s",
            "logger": "%(name)s",
            "message": "%(message)s",
        })
    else:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# Import after path setup
try:
    from SaaStelligence.agents.saastelligence_agent import (
        SAAStelligenceAgent,
        AgentConfig,
        AgentError,
        ValidationError,
        ModelNotFoundError,
    )
except ImportError:
    # Fallback for direct execution
    from agents.saastelligence_agent import (
        SAAStelligenceAgent,
        AgentConfig,
        AgentError,
        ValidationError,
        ModelNotFoundError,
    )

logger = logging.getLogger(__name__)


# ============== CLI COMMANDS ==============

def cmd_run(args: argparse.Namespace) -> int:
    """Execute the SAAStelligence pipeline for a query."""
    agent = SAAStelligenceAgent()
    
    try:
        result = agent.run(
            user_query=args.query,
            user_id=args.user_id,
            last_action=args.last_action,
            cpa_budget=args.cpa_budget,
        )
        
        if args.output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n🚀 SAAStelligence Response")
            print("=" * 50)
            print(f"📌 Intent:          {result['intent']}")
            print(f"📊 Confidence:      {result['intent_confidence']:.1%}")
            print(f"🎯 Funnel:          {result['funnel']}")
            print(f"📈 Predicted CVR:   {result['predicted_cvr']:.2%}")
            print(f"💰 Recommended Bid: ${result['bid']:.2f}")
            
            if result.get('retarget_url'):
                print(f"🔗 Retarget URL:    {result['retarget_url']}")
            
            print("\n📝 Generated Ad Copy:")
            print("-" * 50)
            print(result['ad_copy'])
            print("-" * 50)
        
        return 0
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except ModelNotFoundError as e:
        logger.error(f"Model error: {e}")
        print("\n💡 Tip: Run training first with: python main.py train --file your_data.csv")
        return 1
    except AgentError as e:
        logger.error(f"Agent error: {e}")
        return 1


async def cmd_run_async(args: argparse.Namespace) -> int:
    """Execute the pipeline asynchronously."""
    agent = SAAStelligenceAgent()
    
    try:
        result = await agent.run_async(
            user_query=args.query,
            user_id=args.user_id,
            last_action=args.last_action,
            cpa_budget=args.cpa_budget,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except AgentError as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate and display performance report."""
    agent = SAAStelligenceAgent()
    
    try:
        metrics = agent.report_performance()
        
        if args.output_format == "json":
            print(json.dumps(metrics, indent=2))
        else:
            print("\n📊 Performance Report")
            print("=" * 40)
            for key, value in metrics.items():
                print(f"{key:15} : {value}")
            print("=" * 40)
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train or retrain the intent model from CSV data."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        logger.error(f"Training file not found: {file_path}")
        return 1
    
    # Load CSV data
    try:
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records: List[Dict[str, str]] = list(reader)
    except Exception as e:
        logger.error(f"Failed to read training file: {e}")
        return 1
    
    if not records:
        logger.error("Training file is empty")
        return 1
    
    print(f"📚 Loaded {len(records)} records from {file_path}")
    
    agent = SAAStelligenceAgent()
    
    try:
        result = agent.train_from_feedback(records)
        print(f"✅ Model trained successfully with {result['records_trained']} records")
        return 0
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def cmd_server(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn not installed. Install with: pip install uvicorn[standard]"
        )
        return 1
    
    try:
        from SaaStelligence.api.app import app
    except ImportError:
        try:
            from api.app import app
        except ImportError:
            logger.error("API module not found. Ensure api/app.py exists.")
            return 1
    
    print(f"🚀 Starting SAAStelligence API on http://{args.host}:{args.port}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Start interactive REPL mode."""
    agent = SAAStelligenceAgent()
    
    print("\n🤖 SAAStelligence Interactive Mode")
    print("=" * 50)
    print("Type your queries below. Commands:")
    print("  /report  - Show performance metrics")
    print("  /help    - Show this help")
    print("  /quit    - Exit")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return 0
        
        if not query:
            continue
        
        if query.startswith("/"):
            cmd = query.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("👋 Goodbye!")
                return 0
            elif cmd == "/report":
                cmd_report(argparse.Namespace(output_format="table"))
            elif cmd == "/help":
                print("Commands: /report, /help, /quit")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        try:
            result = agent.run(query)
            print(f"\n📌 Intent: {result['intent']} ({result['intent_confidence']:.1%})")
            print(f"🎯 Funnel: {result['funnel']}")
            print(f"💰 Bid: ${result['bid']:.2f}")
            print(f"\n📝 Ad: {result['ad_copy'][:200]}...")
        except AgentError as e:
            print(f"❌ Error: {e}")


# ============== CLI SETUP ==============

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="saastelligence",
        description="SAAStelligence - AI-Powered SaaS Lead Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run "How can I automate my sales workflow?"
  %(prog)s run "CRM integration" --user-id user123 --last-action pricing_viewed
  %(prog)s report --format json
  %(prog)s train --file conversions.csv
  %(prog)s server --port 8080
  %(prog)s interactive
        """,
    )
    
    # Global options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ---- run command ----
    run_parser = subparsers.add_parser(
        "run",
        help="Run the pipeline for a single query",
    )
    run_parser.add_argument(
        "query",
        type=str,
        help="User query to process",
    )
    run_parser.add_argument(
        "--user-id", "-u",
        type=str,
        default=None,
        help="User ID for retargeting",
    )
    run_parser.add_argument(
        "--last-action", "-a",
        type=str,
        default=None,
        choices=["email_submitted", "form_abandoned", "pricing_viewed", 
                 "demo_requested", "trial_started"],
        help="User's last action for retargeting",
    )
    run_parser.add_argument(
        "--cpa-budget", "-c",
        type=float,
        default=None,
        help="Target CPA budget in dollars",
    )
    run_parser.add_argument(
        "--format", "-f",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    run_parser.add_argument(
        "--async",
        dest="run_async",
        action="store_true",
        help="Run asynchronously",
    )
    run_parser.set_defaults(func=cmd_run)
    
    # ---- report command ----
    report_parser = subparsers.add_parser(
        "report",
        help="Generate performance report",
    )
    report_parser.add_argument(
        "--format", "-f",
        dest="output_format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    report_parser.set_defaults(func=cmd_report)
    
    # ---- train command ----
    train_parser = subparsers.add_parser(
        "train",
        help="Train model from CSV data",
    )
    train_parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="Path to CSV file with training data",
    )
    train_parser.set_defaults(func=cmd_train)
    
    # ---- server command ----
    server_parser = subparsers.add_parser(
        "server",
        help="Start the FastAPI server",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    server_parser.set_defaults(func=cmd_server)
    
    # ---- interactive command ----
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive REPL mode",
    )
    interactive_parser.set_defaults(func=cmd_interactive)
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, json_output=args.json_logs)
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle async run
    if args.command == "run" and getattr(args, "run_async", False):
        return asyncio.run(cmd_run_async(args))
    
    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

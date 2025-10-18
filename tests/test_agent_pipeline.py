import csv
import os
import tempfile
import unittest
from importlib import reload
from pathlib import Path

def _write_dataset(path: Path) -> None:
    rows = [
        {
            "query_text": "Automate workflow for my team",
            "intent": "workflow_automation",
            "converted": 1,
            "clicks": 40,
            "impressions": 1000,
            "cost": 300.0,
        },
        {
            "query_text": "Need help scaling support",
            "intent": "customer_support",
            "converted": 0,
            "clicks": 25,
            "impressions": 800,
            "cost": 200.0,
        },
        {
            "query_text": "Improve sales follow-up",
            "intent": "sales_team_efficiency",
            "converted": 1,
            "clicks": 30,
            "impressions": 950,
            "cost": 275.0,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_text", "intent", "converted", "clicks", "impressions", "cost"],
        )
        writer.writeheader()
        writer.writerows(rows)


class AgentPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dataset_path = Path(self.tmpdir.name) / "conversions.csv"
        self.model_path = Path(self.tmpdir.name) / "intent.json"
        _write_dataset(self.dataset_path)

        self.original_env = {
            "CONVERSIONS_DATA_PATH": os.environ.get("CONVERSIONS_DATA_PATH"),
            "INTENT_MODEL_PATH": os.environ.get("INTENT_MODEL_PATH"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        }
        os.environ["CONVERSIONS_DATA_PATH"] = str(self.dataset_path)
        os.environ["INTENT_MODEL_PATH"] = str(self.model_path)
        os.environ.pop("OPENAI_API_KEY", None)

        from SaaStelligence.config import config as config_module

        reload(config_module)

        from SaaStelligence.models import train_intent_model as train_module

        reload(train_module)

        from SaaStelligence.agents import saastelligence_agent as agent_module

        reload(agent_module)
        self.agent_module = agent_module
        self.agent = agent_module.SAAStelligenceAgent()

    def tearDown(self) -> None:
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_agent_run_returns_expected_keys(self) -> None:
        output = self.agent.run("I want to automate marketing emails")

        self.assertIn(output["intent"], self.agent.intent_mapping)
        self.assertGreater(output["bid"], 0)
        self.assertIn("funnel", output)

    def test_report_performance_formats_metrics(self) -> None:
        report = self.agent.report_performance()

        self.assertEqual(set(report), {"CTR", "CVR", "CPA", "Leads"})
        self.assertIsInstance(report["Leads"], int)
        self.assertIsInstance(report["CTR"], float)
        self.assertIsInstance(report["CVR"], float)
        self.assertIsInstance(report["CPA"], float)


if __name__ == "__main__":
    unittest.main()

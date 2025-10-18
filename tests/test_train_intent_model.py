import tempfile
import unittest
from pathlib import Path

from SaaStelligence.models import train_intent_model as tim


SAMPLE_DATA = [
    {
        "query_text": "Automate workflow for my team",
        "intent": "workflow_automation",
        "converted": 1,
    },
    {
        "query_text": "Need help scaling customer support",
        "intent": "customer_support",
        "converted": 0,
    },
    {
        "query_text": "Improve sales follow-up process",
        "intent": "sales_team_efficiency",
        "converted": 1,
    },
    {
        "query_text": "Project tracking tools",
        "intent": "project_management",
        "converted": 0,
    },
    {
        "query_text": "Automate marketing emails",
        "intent": "marketing_automation",
        "converted": 1,
    },
]


class TrainIntentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.model_path = Path(self.tmpdir.name) / "intent.json"

    def test_train_intent_classifier_persists_model(self) -> None:
        model = tim.train_intent_classifier(records=SAMPLE_DATA, model_path=self.model_path)

        self.assertTrue(self.model_path.exists(), "Trained model should be persisted to disk")

        probabilities = model.predict_proba("Need workflow automation tools")
        self.assertEqual(len(probabilities), len(tim.INTENT_CATEGORIES))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)

    def test_map_intents_to_labels_respects_defined_categories(self) -> None:
        labels = tim.map_intents_to_labels(["workflow_automation", "customer_support"])
        self.assertEqual(labels, [0, 3])


if __name__ == "__main__":
    unittest.main()

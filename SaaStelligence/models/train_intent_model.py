# models/train_intent_model.py

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Define intent categories
INTENT_CATEGORIES = [
    "workflow_automation",
    "sales_team_efficiency",
    "project_management",
    "customer_support",
    "marketing_automation",
]

REQUIRED_DATA_COLUMNS = {"query_text", "intent", "converted"}

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "conversions.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "intent_classifier.json"


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def load_data(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Load the canonical conversions dataset and validate its schema."""

    dataset_path = Path(path) if path else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Provide a CSV file with the required columns: "
            f"{sorted(REQUIRED_DATA_COLUMNS)}."
        )

    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_DATA_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Dataset at {dataset_path} is missing required columns: {sorted(missing)}."
            )
        records = [row for row in reader if row.get("query_text") and row.get("intent")]
    return records


def map_intents_to_labels(intents: Iterable[str]) -> List[int]:
    """Convert string intents to numeric class indices."""

    intent_to_label = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
    labels: List[int] = []
    for intent in intents:
        if intent not in intent_to_label:
            raise ValueError(
                f"Encountered unknown intent '{intent}'. Valid intents: {INTENT_CATEGORIES}"
            )
        labels.append(intent_to_label[intent])
    return labels


class NaiveBayesIntentClassifier:
    """Simple multinomial Naive Bayes classifier for intent detection."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.label_word_counts: Dict[int, Counter[str]] = defaultdict(Counter)
        self.label_counts: Counter[int] = Counter()
        self.total_words: Counter[int] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        for text, label in zip(texts, labels):
            tokens = _tokenize(text)
            if not tokens:
                continue
            self.label_counts[label] += 1
            for token in tokens:
                self.label_word_counts[label][token] += 1
                self.total_words[label] += 1
                self.vocabulary.add(token)
        # Ensure every label is represented to avoid zero-prior collapse.
        for label in range(len(INTENT_CATEGORIES)):
            self.label_counts[label] += 0
            self.total_words[label] += 0

    def predict_proba(self, text: str) -> List[float]:
        tokens = _tokenize(text)
        if not tokens:
            # Return uniform distribution if no signal is present.
            return [1.0 / len(INTENT_CATEGORIES)] * len(INTENT_CATEGORIES)

        total_documents = sum(self.label_counts.values())
        vocab_size = max(len(self.vocabulary), 1)
        log_probabilities: List[float] = []

        for label in range(len(INTENT_CATEGORIES)):
            prior_numerator = self.label_counts[label] + self.alpha
            prior_denominator = total_documents + self.alpha * len(INTENT_CATEGORIES)
            log_prob = math.log(prior_numerator / prior_denominator)

            for token in tokens:
                token_count = self.label_word_counts[label][token]
                word_numerator = token_count + self.alpha
                word_denominator = self.total_words[label] + self.alpha * vocab_size
                log_prob += math.log(word_numerator / word_denominator)

            log_probabilities.append(log_prob)

        max_log = max(log_probabilities)
        exp_probs = [math.exp(log_prob - max_log) for log_prob in log_probabilities]
        total = sum(exp_probs) or 1.0
        return [prob / total for prob in exp_probs]

    def predict(self, text: str) -> int:
        probabilities = self.predict_proba(text)
        return max(range(len(probabilities)), key=lambda idx: probabilities[idx])

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "label_counts": dict(self.label_counts),
            "total_words": dict(self.total_words),
            "label_word_counts": {
                str(label): dict(counter) for label, counter in self.label_word_counts.items()
            },
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NaiveBayesIntentClassifier":
        classifier = cls(alpha=float(payload.get("alpha", 1.0)))
        classifier.label_counts = Counter({int(k): int(v) for k, v in payload.get("label_counts", {}).items()})
        classifier.total_words = Counter({int(k): int(v) for k, v in payload.get("total_words", {}).items()})
        classifier.label_word_counts = defaultdict(
            Counter,
            {
                int(label): Counter({token: int(count) for token, count in counter.items()})
                for label, counter in payload.get("label_word_counts", {}).items()
            },
        )
        classifier.vocabulary = set(payload.get("vocabulary", []))
        return classifier


def build_and_train_model(texts: Iterable[str], labels: Sequence[int]) -> NaiveBayesIntentClassifier:
    classifier = NaiveBayesIntentClassifier()
    classifier.fit(list(texts), list(labels))
    return classifier


def save_model(model: NaiveBayesIntentClassifier, model_path: Optional[Path] = None) -> Path:
    destination = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle)
    return destination


def load_model(model_path: Optional[Path] = None) -> NaiveBayesIntentClassifier:
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return NaiveBayesIntentClassifier.from_dict(payload)


def train_intent_classifier(
    records: Optional[List[Dict[str, str]]] = None,
    model_path: Optional[Path] = None,
) -> NaiveBayesIntentClassifier:
    dataset = records if records is not None else load_data()
    texts = [record["query_text"] for record in dataset]
    labels = map_intents_to_labels(record["intent"] for record in dataset)
    model = build_and_train_model(texts, labels)
    save_model(model, model_path=model_path)
    return model


if __name__ == "__main__":
    train_intent_classifier()
    print("✅ Model saved to the 'models/' directory.")

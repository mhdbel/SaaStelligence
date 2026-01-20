"""
Intent classification model training and inference.

This module provides a lightweight Naive Bayes classifier for intent detection
that requires no external ML dependencies. The model is serialized to JSON
for easy deployment and portability.

Features:
- Multinomial Naive Bayes with Laplace smoothing
- Log-space probability calculations (prevents underflow)
- Text preprocessing with configurable options
- Model versioning and metadata
- Validation metrics during training
- Thread-safe inference

Example:
    >>> from models.train_intent_model import train_intent_classifier, load_model
    >>> 
    >>> # Train a new model
    >>> model = train_intent_classifier()
    >>> 
    >>> # Load and use existing model
    >>> model = load_model()
    >>> probs = model.predict_proba("automate my sales workflow")
    >>> intent_idx = model.predict("automate my sales workflow")
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# ============== LOGGING ==============
logger = logging.getLogger(__name__)

# ============== CONSTANTS ==============
MODEL_VERSION = "1.1.0"

INTENT_CATEGORIES: List[str] = [
    "workflow_automation",
    "sales_team_efficiency",
    "project_management",
    "customer_support",
    "marketing_automation",
]

REQUIRED_DATA_COLUMNS: FrozenSet[str] = frozenset({
    "query_text",
    "intent",
    "converted",
})

# Common English stopwords (lightweight set)
DEFAULT_STOPWORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "as", "if", "then", "than", "so", "just",
    "now", "here", "there", "when", "where", "why", "how", "all", "each",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "too", "very", "just", "also",
})

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "conversions.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "intent_classifier.json"

# Limits
MAX_VOCABULARY_SIZE = 50_000
MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 50
MAX_TOKENS_PER_DOCUMENT = 500


# ============== EXCEPTIONS ==============
class ModelError(Exception):
    """Base exception for model errors."""
    pass


class ModelVersionError(ModelError):
    """Raised when model version is incompatible."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when trying to predict with untrained model."""
    pass


class DataValidationError(ModelError):
    """Raised when training data is invalid."""
    pass


# ============== DATA CLASSES ==============
@dataclass
class TrainingMetrics:
    """Metrics from model training."""
    total_records: int = 0
    valid_records: int = 0
    skipped_records: int = 0
    vocabulary_size: int = 0
    records_per_intent: Dict[str, int] = field(default_factory=dict)
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_time_seconds: float = 0.0


@dataclass
class PredictionResult:
    """Result of a prediction with confidence information."""
    intent: str
    intent_index: int
    confidence: float
    probabilities: Dict[str, float]
    is_confident: bool
    
    @classmethod
    def from_probabilities(
        cls,
        probabilities: List[float],
        confidence_threshold: float = 0.3,
    ) -> "PredictionResult":
        """Create PredictionResult from probability list."""
        intent_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
        confidence = probabilities[intent_index]
        intent = INTENT_CATEGORIES[intent_index]
        
        return cls(
            intent=intent,
            intent_index=intent_index,
            confidence=confidence,
            probabilities={
                INTENT_CATEGORIES[i]: prob 
                for i, prob in enumerate(probabilities)
            },
            is_confident=confidence >= confidence_threshold,
        )


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""
    version: str
    created_at: str
    intent_categories: List[str]
    training_records: int
    vocabulary_size: int
    alpha: float
    preprocessing_config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "intent_categories": self.intent_categories,
            "training_records": self.training_records,
            "vocabulary_size": self.vocabulary_size,
            "alpha": self.alpha,
            "preprocessing_config": self.preprocessing_config,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            version=data.get("version", "unknown"),
            created_at=data.get("created_at", "unknown"),
            intent_categories=data.get("intent_categories", INTENT_CATEGORIES),
            training_records=data.get("training_records", 0),
            vocabulary_size=data.get("vocabulary_size", 0),
            alpha=data.get("alpha", 1.0),
            preprocessing_config=data.get("preprocessing_config", {}),
            metrics=data.get("metrics"),
        )


# ============== TEXT PREPROCESSING ==============
@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = True
    min_token_length: int = MIN_TOKEN_LENGTH
    max_token_length: int = MAX_TOKEN_LENGTH
    max_tokens: int = MAX_TOKENS_PER_DOCUMENT
    stopwords: FrozenSet[str] = DEFAULT_STOPWORDS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_numbers": self.remove_numbers,
            "remove_stopwords": self.remove_stopwords,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "max_tokens": self.max_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(
            lowercase=data.get("lowercase", True),
            remove_punctuation=data.get("remove_punctuation", True),
            remove_numbers=data.get("remove_numbers", False),
            remove_stopwords=data.get("remove_stopwords", True),
            min_token_length=data.get("min_token_length", MIN_TOKEN_LENGTH),
            max_token_length=data.get("max_token_length", MAX_TOKEN_LENGTH),
            max_tokens=data.get("max_tokens", MAX_TOKENS_PER_DOCUMENT),
        )


class TextPreprocessor:
    """Text preprocessing for intent classification."""
    
    # Regex patterns (compiled once)
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    NUMBERS_PATTERN = re.compile(r'\d+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text.
        
        Args:
            text: Raw input text.
            
        Returns:
            List of preprocessed tokens.
        """
        if not text or not isinstance(text, str):
            return []
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = self.PUNCTUATION_PATTERN.sub(' ', text)
        
        # Remove numbers
        if self.config.remove_numbers:
            text = self.NUMBERS_PATTERN.sub(' ', text)
        
        # Normalize whitespace and split
        text = self.WHITESPACE_PATTERN.sub(' ', text).strip()
        tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Length filter
            if len(token) < self.config.min_token_length:
                continue
            if len(token) > self.config.max_token_length:
                continue
            
            # Stopword filter
            if self.config.remove_stopwords and token in self.config.stopwords:
                continue
            
            filtered_tokens.append(token)
            
            # Max tokens limit
            if len(filtered_tokens) >= self.config.max_tokens:
                break
        
        return filtered_tokens


# Default preprocessor instance
_default_preprocessor = TextPreprocessor()


def tokenize(text: str, preprocessor: Optional[TextPreprocessor] = None) -> List[str]:
    """
    Tokenize text using the specified or default preprocessor.
    
    Args:
        text: Raw input text.
        preprocessor: Optional custom preprocessor.
        
    Returns:
        List of tokens.
    """
    proc = preprocessor or _default_preprocessor
    return proc.tokenize(text)


# ============== DATA LOADING ==============
def load_data(
    path: Optional[Path] = None,
    validate: bool = True,
) -> List[Dict[str, str]]:
    """
    Load the conversions dataset from CSV.
    
    Args:
        path: Path to CSV file. Defaults to DEFAULT_DATA_PATH.
        validate: Whether to validate the schema.
        
    Returns:
        List of record dictionaries.
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        DataValidationError: If schema validation fails.
    """
    dataset_path = Path(path) if path else DEFAULT_DATA_PATH
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Provide a CSV file with columns: {sorted(REQUIRED_DATA_COLUMNS)}"
        )
    
    records: List[Dict[str, str]] = []
    skipped = 0
    
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            
            # Validate schema
            if validate:
                fieldnames = set(reader.fieldnames or [])
                missing = REQUIRED_DATA_COLUMNS - fieldnames
                if missing:
                    raise DataValidationError(
                        f"Dataset missing required columns: {sorted(missing)}. "
                        f"Found columns: {sorted(fieldnames)}"
                    )
            
            # Read records
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                query = row.get("query_text", "").strip()
                intent = row.get("intent", "").strip()
                
                if not query or not intent:
                    skipped += 1
                    logger.debug(f"Skipping row {row_num}: empty query or intent")
                    continue
                
                if intent not in INTENT_CATEGORIES:
                    skipped += 1
                    logger.warning(
                        f"Skipping row {row_num}: unknown intent '{intent}'"
                    )
                    continue
                
                records.append(row)
    
    except csv.Error as e:
        raise DataValidationError(f"CSV parsing error: {e}")
    
    if skipped > 0:
        logger.info(f"Loaded {len(records)} records, skipped {skipped} invalid rows")
    else:
        logger.info(f"Loaded {len(records)} records from {dataset_path}")
    
    return records


def map_intents_to_labels(intents: Iterable[str]) -> List[int]:
    """
    Convert string intents to numeric class indices.
    
    Args:
        intents: Iterable of intent strings.
        
    Returns:
        List of integer labels.
        
    Raises:
        ValueError: If unknown intent is encountered.
    """
    intent_to_label = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
    labels: List[int] = []
    
    for intent in intents:
        intent = intent.strip()
        if intent not in intent_to_label:
            raise ValueError(
                f"Unknown intent '{intent}'. Valid intents: {INTENT_CATEGORIES}"
            )
        labels.append(intent_to_label[intent])
    
    return labels


# ============== NAIVE BAYES CLASSIFIER ==============
class NaiveBayesIntentClassifier:
    """
    Multinomial Naive Bayes classifier for intent detection.
    
    This implementation uses:
    - Laplace (additive) smoothing to handle unseen words
    - Log-space probability calculations to prevent underflow
    - Vocabulary size limiting to control memory usage
    
    Thread Safety:
        The predict() and predict_proba() methods are thread-safe.
        The fit() method is NOT thread-safe and should only be called
        during model training.
    
    Example:
        >>> classifier = NaiveBayesIntentClassifier()
        >>> classifier.fit(["automate workflow", "manage projects"], [0, 2])
        >>> probs = classifier.predict_proba("workflow automation")
        >>> print(probs)
        [0.65, 0.1, 0.15, 0.05, 0.05]
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        max_vocabulary_size: int = MAX_VOCABULARY_SIZE,
        preprocessor: Optional[TextPreprocessor] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the classifier.
        
        Args:
            alpha: Laplace smoothing parameter. Higher values = more smoothing.
            max_vocabulary_size: Maximum vocabulary size to prevent OOM.
            preprocessor: Text preprocessor instance.
            confidence_threshold: Minimum confidence for "confident" predictions.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        self.alpha = alpha
        self.max_vocabulary_size = max_vocabulary_size
        self.preprocessor = preprocessor or TextPreprocessor()
        self.confidence_threshold = confidence_threshold
        
        # Model state
        self.label_word_counts: Dict[int, Counter[str]] = defaultdict(Counter)
        self.label_counts: Counter[int] = Counter()
        self.total_words: Counter[int] = Counter()
        self.vocabulary: Set[str] = set()
        
        # Metadata
        self._is_trained = False
        self._training_records = 0
        self._created_at: Optional[str] = None
        
        # Thread safety for inference
        self._lock = threading.RLock()
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    @property
    def vocabulary_size(self) -> int:
        """Get the current vocabulary size."""
        return len(self.vocabulary)
    
    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        prune_vocabulary: bool = True,
    ) -> TrainingMetrics:
        """
        Train the classifier on the provided data.
        
        Args:
            texts: Sequence of text documents.
            labels: Sequence of integer labels (same length as texts).
            prune_vocabulary: Whether to prune vocabulary to max size.
            
        Returns:
            TrainingMetrics with training statistics.
            
        Raises:
            ValueError: If texts and labels have different lengths.
        """
        import time
        start_time = time.time()
        
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have same length: {len(texts)} != {len(labels)}"
            )
        
        # Reset state
        self.label_word_counts = defaultdict(Counter)
        self.label_counts = Counter()
        self.total_words = Counter()
        self.vocabulary = set()
        
        # Track metrics
        valid_records = 0
        skipped_records = 0
        records_per_intent: Counter[str] = Counter()
        
        # Collect all word counts
        word_document_frequency: Counter[str] = Counter()
        
        for text, label in zip(texts, labels):
            tokens = self.preprocessor.tokenize(text)
            
            if not tokens:
                skipped_records += 1
                continue
            
            valid_records += 1
            self.label_counts[label] += 1
            records_per_intent[INTENT_CATEGORIES[label]] += 1
            
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_document_frequency[token] += 1
            
            for token in tokens:
                self.label_word_counts[label][token] += 1
                self.total_words[label] += 1
                self.vocabulary.add(token)
        
        # Prune vocabulary if needed
        if prune_vocabulary and len(self.vocabulary) > self.max_vocabulary_size:
            logger.info(
                f"Pruning vocabulary from {len(self.vocabulary)} to {self.max_vocabulary_size}"
            )
            
            # Keep most frequent words
            top_words = word_document_frequency.most_common(self.max_vocabulary_size)
            self.vocabulary = {word for word, _ in top_words}
            
            # Remove pruned words from counts
            for label in self.label_word_counts:
                pruned_counter = Counter({
                    word: count 
                    for word, count in self.label_word_counts[label].items()
                    if word in self.vocabulary
                })
                removed_count = self.total_words[label] - sum(pruned_counter.values())
                self.label_word_counts[label] = pruned_counter
                self.total_words[label] -= removed_count
        
        # Ensure all labels are represented (prevent zero-prior issues)
        for label in range(len(INTENT_CATEGORIES)):
            if label not in self.label_counts:
                self.label_counts[label] = 0
                self.total_words[label] = 0
        
        self._is_trained = True
        self._training_records = valid_records
        self._created_at = datetime.utcnow().isoformat() + "Z"
        
        training_time = time.time() - start_time
        
        metrics = TrainingMetrics(
            total_records=len(texts),
            valid_records=valid_records,
            skipped_records=skipped_records,
            vocabulary_size=len(self.vocabulary),
            records_per_intent=dict(records_per_intent),
            training_time_seconds=round(training_time, 3),
        )
        
        logger.info(
            f"Training complete: {valid_records} records, "
            f"{len(self.vocabulary)} vocabulary, "
            f"{training_time:.2f}s"
        )
        
        return metrics
    
    def predict_proba(self, text: str) -> List[float]:
        """
        Predict class probabilities for the given text.
        
        Thread-safe.
        
        Args:
            text: Input text to classify.
            
        Returns:
            List of probabilities, one per intent category.
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained.
        """
        with self._lock:
            if not self._is_trained:
                raise ModelNotTrainedError(
                    "Model has not been trained. Call fit() first or load a trained model."
                )
            
            tokens = self.preprocessor.tokenize(text)
            
            # Return uniform distribution if no tokens
            if not tokens:
                num_classes = len(INTENT_CATEGORIES)
                return [1.0 / num_classes] * num_classes
            
            total_documents = sum(self.label_counts.values())
            vocab_size = max(len(self.vocabulary), 1)
            num_classes = len(INTENT_CATEGORIES)
            
            log_probabilities: List[float] = []
            
            for label in range(num_classes):
                # Prior probability with smoothing
                prior_num = self.label_counts[label] + self.alpha
                prior_denom = total_documents + self.alpha * num_classes
                log_prob = math.log(prior_num / prior_denom)
                
                # Likelihood for each token
                for token in tokens:
                    if token not in self.vocabulary:
                        # Skip OOV tokens (alternatively could use UNK handling)
                        continue
                    
                    token_count = self.label_word_counts[label].get(token, 0)
                    word_num = token_count + self.alpha
                    word_denom = self.total_words[label] + self.alpha * vocab_size
                    log_prob += math.log(word_num / word_denom)
                
                log_probabilities.append(log_prob)
            
            # Convert from log-space with numerical stability
            max_log = max(log_probabilities)
            exp_probs = [math.exp(lp - max_log) for lp in log_probabilities]
            total = sum(exp_probs) or 1.0
            
            return [prob / total for prob in exp_probs]
    
    def predict(self, text: str) -> int:
        """
        Predict the most likely class for the given text.
        
        Thread-safe.
        
        Args:
            text: Input text to classify.
            
        Returns:
            Integer label of the most likely class.
        """
        probabilities = self.predict_proba(text)
        return max(range(len(probabilities)), key=lambda i: probabilities[i])
    
    def predict_intent(self, text: str) -> PredictionResult:
        """
        Predict intent with full result information.
        
        Thread-safe.
        
        Args:
            text: Input text to classify.
            
        Returns:
            PredictionResult with intent, confidence, and probabilities.
        """
        probabilities = self.predict_proba(text)
        return PredictionResult.from_probabilities(
            probabilities,
            confidence_threshold=self.confidence_threshold,
        )
    
    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> Dict[str, float]:
        """
        Evaluate the model on a test set.
        
        Args:
            texts: Test documents.
            labels: True labels.
            
        Returns:
            Dictionary with accuracy and per-class metrics.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        
        correct = 0
        total = 0
        per_class_correct: Counter[int] = Counter()
        per_class_total: Counter[int] = Counter()
        
        for text, true_label in zip(texts, labels):
            predicted = self.predict(text)
            per_class_total[true_label] += 1
            total += 1
            
            if predicted == true_label:
                correct += 1
                per_class_correct[true_label] += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        per_class_accuracy = {}
        for label in range(len(INTENT_CATEGORIES)):
            class_total = per_class_total[label]
            if class_total > 0:
                per_class_accuracy[INTENT_CATEGORIES[label]] = (
                    per_class_correct[label] / class_total
                )
        
        return {
            "accuracy": round(accuracy, 4),
            "total_samples": total,
            "correct_predictions": correct,
            "per_class_accuracy": per_class_accuracy,
        }
    
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the model to a dictionary.
        
        Returns:
            Dictionary representation of the model.
        """
        return {
            "metadata": ModelMetadata(
                version=MODEL_VERSION,
                created_at=self._created_at or datetime.utcnow().isoformat() + "Z",
                intent_categories=INTENT_CATEGORIES,
                training_records=self._training_records,
                vocabulary_size=len(self.vocabulary),
                alpha=self.alpha,
                preprocessing_config=self.preprocessor.config.to_dict(),
            ).to_dict(),
            "model": {
                "alpha": self.alpha,
                "confidence_threshold": self.confidence_threshold,
                "label_counts": dict(self.label_counts),
                "total_words": dict(self.total_words),
                "label_word_counts": {
                    str(label): dict(counter)
                    for label, counter in self.label_word_counts.items()
                },
                "vocabulary": sorted(self.vocabulary),
            },
        }
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NaiveBayesIntentClassifier":
        """
        Deserialize a model from a dictionary.
        
        Args:
            payload: Dictionary representation of the model.
            
        Returns:
            Reconstructed classifier instance.
            
        Raises:
            ModelVersionError: If model version is incompatible.
        """
        # Handle both old and new formats
        if "metadata" in payload:
            metadata = ModelMetadata.from_dict(payload["metadata"])
            model_data = payload.get("model", payload)
            
            # Version check
            model_version = metadata.version
            if model_version.split(".")[0] != MODEL_VERSION.split(".")[0]:
                raise ModelVersionError(
                    f"Model version {model_version} is incompatible with "
                    f"current version {MODEL_VERSION}"
                )
            
            preprocessing_config = PreprocessingConfig.from_dict(
                metadata.preprocessing_config
            )
        else:
            # Legacy format
            model_data = payload
            preprocessing_config = PreprocessingConfig()
            metadata = None
        
        # Create classifier
        classifier = cls(
            alpha=float(model_data.get("alpha", 1.0)),
            preprocessor=TextPreprocessor(preprocessing_config),
            confidence_threshold=float(model_data.get("confidence_threshold", 0.3)),
        )
        
        # Restore state
        classifier.label_counts = Counter({
            int(k): int(v)
            for k, v in model_data.get("label_counts", {}).items()
        })
        
        classifier.total_words = Counter({
            int(k): int(v)
            for k, v in model_data.get("total_words", {}).items()
        })
        
        classifier.label_word_counts = defaultdict(
            Counter,
            {
                int(label): Counter({
                    token: int(count)
                    for token, count in counter.items()
                })
                for label, counter in model_data.get("label_word_counts", {}).items()
            },
        )
        
        classifier.vocabulary = set(model_data.get("vocabulary", []))
        classifier._is_trained = len(classifier.vocabulary) > 0
        
        if metadata:
            classifier._training_records = metadata.training_records
            classifier._created_at = metadata.created_at
        
        return classifier


# ============== MODEL TRAINING AND PERSISTENCE ==============

def build_and_train_model(
    texts: Sequence[str],
    labels: Sequence[int],
    alpha: float = 1.0,
    preprocessor: Optional[TextPreprocessor] = None,
) -> Tuple[NaiveBayesIntentClassifier, TrainingMetrics]:
    """
    Build and train a new classifier.
    
    Args:
        texts: Training documents.
        labels: Training labels.
        alpha: Smoothing parameter.
        preprocessor: Optional custom preprocessor.
        
    Returns:
        Tuple of (trained classifier, training metrics).
    """
    classifier = NaiveBayesIntentClassifier(
        alpha=alpha,
        preprocessor=preprocessor,
    )
    metrics = classifier.fit(texts, labels)
    return classifier, metrics


def save_model(
    model: NaiveBayesIntentClassifier,
    model_path: Optional[Path] = None,
) -> Path:
    """
    Save a trained model to JSON file.
    
    Args:
        model: Trained classifier instance.
        model_path: Destination path. Defaults to DEFAULT_MODEL_PATH.
        
    Returns:
        Path where model was saved.
    """
    destination = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle, indent=2)
    
    logger.info(f"Model saved to {destination}")
    return destination


def load_model(model_path: Optional[Path] = None) -> NaiveBayesIntentClassifier:
    """
    Load a trained model from JSON file.
    
    Args:
        model_path: Path to model file. Defaults to DEFAULT_MODEL_PATH.
        
    Returns:
        Loaded classifier instance.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ModelVersionError: If model version is incompatible.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    
    model = NaiveBayesIntentClassifier.from_dict(payload)
    logger.info(f"Model loaded from {path}")
    
    return model


def train_intent_classifier(
    records: Optional[List[Dict[str, str]]] = None,
    model_path: Optional[Path] = None,
    test_split: float = 0.0,
    alpha: float = 1.0,
) -> NaiveBayesIntentClassifier:
    """
    Train and save an intent classifier.
    
    This is the main entry point for model training.
    
    Args:
        records: Training records. If None, loads from default path.
        model_path: Where to save the model. Defaults to DEFAULT_MODEL_PATH.
        test_split: Fraction of data to hold out for testing (0.0 to 0.5).
        alpha: Smoothing parameter.
        
    Returns:
        Trained classifier instance.
    """
    # Load data
    dataset = records if records is not None else load_data()
    
    if not dataset:
        raise DataValidationError("No training data available")
    
    # Extract texts and labels
    texts = [record["query_text"] for record in dataset]
    labels = map_intents_to_labels(record["intent"] for record in dataset)
    
    # Optional train/test split
    if 0.0 < test_split <= 0.5:
        split_idx = int(len(texts) * (1 - test_split))
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        logger.info(
            f"Split data: {len(train_texts)} train, {len(test_texts)} test"
        )
    else:
        train_texts, train_labels = texts, labels
        test_texts, test_labels = [], []
    
    # Train model
    model, metrics = build_and_train_model(train_texts, train_labels, alpha=alpha)
    
    # Evaluate if we have test data
    if test_texts:
        eval_results = model.evaluate(test_texts, test_labels)
        metrics.validation_accuracy = eval_results["accuracy"]
        logger.info(f"Validation accuracy: {eval_results['accuracy']:.1%}")
    
    # Calculate training accuracy
    if train_texts:
        train_eval = model.evaluate(train_texts, train_labels)
        metrics.training_accuracy = train_eval["accuracy"]
        logger.info(f"Training accuracy: {train_eval['accuracy']:.1%}")
    
    # Save model
    save_model(model, model_path=model_path)
    
    return model


# ============== CLI ENTRY POINT ==============

def main() -> None:
    """Command-line entry point for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train the SAAStelligence intent classifier"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to training data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Smoothing parameter (default: 1.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    try:
        records = load_data(args.data)
        model = train_intent_classifier(
            records=records,
            model_path=args.output,
            test_split=args.test_split,
            alpha=args.alpha,
        )
        
        print(f"\n✅ Model trained successfully!")
        print(f"   📁 Saved to: {args.output}")
        print(f"   📊 Vocabulary size: {model.vocabulary_size}")
        print(f"   📝 Training records: {model._training_records}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# models/train_intent_model.py

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Define intent categories
INTENT_CATEGORIES = [
    "workflow_automation",
    "sales_team_efficiency",
    "project_management",
    "customer_support",
    "marketing_automation",
]

REQUIRED_DATA_COLUMNS = {"query_text", "intent", "converted"}
MAX_SEQUENCE_LENGTH = 20
VOCABULARY_SIZE = 5000

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "conversions.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "intent_classifier.h5"
DEFAULT_TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the canonical conversions dataset and validate its schema."""

    dataset_path = Path(path) if path else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Provide a CSV file with the required columns: "
            f"{sorted(REQUIRED_DATA_COLUMNS)}."
        )

    df = pd.read_csv(dataset_path)
    missing = REQUIRED_DATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {dataset_path} is missing required columns: {sorted(missing)}."
        )

    df = df.dropna(subset=["query_text", "intent"]).copy()
    df["query_text"] = df["query_text"].astype(str)
    df["intent"] = df["intent"].astype(str)
    df["converted"] = df["converted"].astype(int)
    return df


def preprocess_text(
    texts: Iterable[str],
    max_length: int = MAX_SEQUENCE_LENGTH,
    num_words: int = VOCABULARY_SIZE,
) -> Tuple[np.ndarray, Tokenizer, int]:
    """Fit a tokenizer on the provided texts and return padded sequences."""

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    return padded, tokenizer, vocab_size


def map_intents_to_labels(intents: Iterable[str]) -> np.ndarray:
    """Convert string intents to numeric class indices."""

    intent_to_label = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
    try:
        labels = [intent_to_label[i] for i in intents]
    except KeyError as exc:
        raise ValueError(
            f"Encountered unknown intent '{exc.args[0]}'. Valid intents: {INTENT_CATEGORIES}"
        ) from exc
    return np.array(labels, dtype=np.int64)


def build_and_train_model(
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> Sequential:
    """Construct and fit the intent classification model."""

    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length),
            GlobalAveragePooling1D(),
            Dense(24, activation="relu"),
            Dense(len(INTENT_CATEGORIES), activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(X, y, epochs=10, validation_split=0.2, verbose=0)
    return model


def save_artifacts(
    model: Sequential,
    tokenizer: Tokenizer,
    model_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> None:
    """Persist the trained model and tokenizer to disk."""

    model_destination = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    tokenizer_destination = (
        Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER_PATH
    )
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_destination.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_destination)
    artifact = {
        "tokenizer_config": tokenizer.to_json(),
        "max_sequence_length": max_length,
    }
    with open(tokenizer_destination, "w", encoding="utf-8") as file:
        json.dump(artifact, file)


def train_intent_classifier(
    dataframe: Optional[pd.DataFrame] = None,
    model_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
) -> Tuple[Sequential, Tokenizer]:
    """Train the intent classifier using the provided dataframe or the default dataset."""

    df = dataframe if dataframe is not None else load_data()
    texts = df["query_text"].astype(str).tolist()
    labels = map_intents_to_labels(df["intent"].tolist())
    sequences, tokenizer, vocab_size = preprocess_text(texts)
    model = build_and_train_model(sequences, labels, vocab_size)
    save_artifacts(model, tokenizer, model_path, tokenizer_path)
    return model, tokenizer


if __name__ == "__main__":
    train_intent_classifier()
    print("✅ Model and tokenizer saved to the 'models/' directory.")
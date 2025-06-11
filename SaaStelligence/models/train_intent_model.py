# models/train_intent_model.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Define intent categories
INTENT_CATEGORIES = [
    'workflow_automation',
    'sales_team_efficiency',
    'project_management',
    'customer_support',
    'marketing_automation'
]

# Load training data
def load_data():
    df = pd.read_csv('data/conversions.csv')
    return df['intent'], df['converted']

# Preprocess text
def preprocess_text(texts):
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')
    return padded, tokenizer

# Map intent to label
def map_intents_to_labels(intents):
    intent_to_label = {intent: idx for idx, intent in enumerate(INTENT_CATEGORIES)}
    labels = [intent_to_label[i] for i in intents]
    return np.array(labels)

# Build and train model
def build_and_train_model(X, y):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=16, input_length=20),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(len(INTENT_CATEGORIES), activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    model.fit(X, y, epochs=10, validation_split=0.2)
    return model

# Save model
def save_model(model, tokenizer, model_path="models/intent_classifier.h5", tokenizer_path="models/tokenizer.json"):
    model.save(model_path)
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.word_index, f)

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    texts, _ = load_data()
    X, tokenizer = preprocess_text(texts)
    y = map_intents_to_labels(texts)

    # Step 2: Train model
    model = build_and_train_model(X, y)

    # Step 3: Save model and tokenizer
    if not os.path.exists("models"):
        os.makedirs("models")
    save_model(model, tokenizer)
    print("✅ Model and tokenizer saved to 'models/' directory.")
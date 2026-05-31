"""
train.py

Train LightGBM model and save artifacts.
"""

import os
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def train():

    print("Loading data...")

    loader = DataLoader(DATA_DIR)

    df = loader.load_application_train()

    target_col = "TARGET"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("Preprocessing data...")

    processor = DataPreprocessor()

    X = processor.preprocess(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training LightGBM...")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    valid_probs = model.predict_proba(X_valid)[:, 1]
    valid_preds = model.predict(X_valid)

    auc = roc_auc_score(
        y_valid,
        valid_probs
    )

    acc = accuracy_score(
        y_valid,
        valid_preds
    )

    print(f"ROC AUC : {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    joblib.dump(
        model,
        f"{MODEL_DIR}/saved_model.pkl"
    )

    joblib.dump(
        processor,
        f"{MODEL_DIR}/preprocessor.pkl"
    )

    print("Artifacts saved successfully")


if __name__ == "__main__":
    train()

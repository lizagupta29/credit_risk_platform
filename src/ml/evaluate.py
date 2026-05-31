"""
evaluate.py

Model evaluation metrics.
"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


MODEL_PATH = "models/saved_model.pkl"


def evaluate():

    loader = DataLoader("data")

    df = loader.load_application_train()

    X = df.drop(
        columns=["TARGET"]
    )

    y = df["TARGET"]

    processor = DataPreprocessor()

    X = processor.preprocess(X)

    model = joblib.load(
        MODEL_PATH
    )

    preds = model.predict(X)

    probs = model.predict_proba(X)[:, 1]

    print("\nClassification Report\n")

    print(
        classification_report(
            y,
            preds
        )
    )

    print("\nConfusion Matrix\n")

    cm = confusion_matrix(
        y,
        preds
    )

    print(cm)

    auc = roc_auc_score(
        y,
        probs
    )

    print(
        f"\nROC-AUC: {auc:.4f}"
    )

    plot_confusion_matrix(cm)

    plot_roc_curve(
        y,
        probs
    )

    plot_precision_recall(
        y,
        probs
    )


def plot_confusion_matrix(cm):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.savefig(
        "documents/confusion_matrix.png"
    )

    plt.show()


def plot_roc_curve(
        y_true,
        probabilities
):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label="ROC Curve"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.savefig(
        "documents/roc_curve.png"
    )

    plt.show()


def plot_precision_recall(
        y_true,
        probabilities
):

    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision Recall Curve"
    )

    plt.savefig(
        "documents/pr_curve.png"
    )

    plt.show()


if __name__ == "__main__":
    evaluate()

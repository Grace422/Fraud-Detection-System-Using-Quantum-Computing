import sys
from pathlib import Path

# Allow importing preprocessing.py
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import prepare_data

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def train_model():

    print("Preparing data...")

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training completed.")

    # Predictions
    y_pred = model.predict(X_test)

    # Fraud probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_probability)

    matrix = confusion_matrix(y_test, y_pred)

    print("\n========== CLASSICAL BASELINE ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    return model


if __name__ == "__main__":
    train_model()
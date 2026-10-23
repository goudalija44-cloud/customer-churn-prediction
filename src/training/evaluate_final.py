
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_cv_model.joblib"
RESULTS_PATH = ROOT_DIR / "data" / "processed" / "final_evaluation.csv"

RANDOM_STATE = 42
THRESHOLD = 0.34


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    results = {
        "Model": [type(model.named_steps["classifier"]).__name__],
        "Threshold": [THRESHOLD],
        "Accuracy": [accuracy_score(y_test, predictions)],
        "Precision": [
            precision_score(y_test, predictions, zero_division=0)
        ],
        "Recall": [recall_score(y_test, predictions, zero_division=0)],
        "F1": [f1_score(y_test, predictions, zero_division=0)],
        "ROC_AUC": [roc_auc_score(y_test, probabilities)],
        "PR_AUC": [average_precision_score(y_test, probabilities)],
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n===== FINAL TEST-SET EVALUATION =====")
    print(results_df.round(4).to_string(index=False))

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("\n===== CONFUSION MATRIX COUNTS =====")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    print(f"\nSaved final metrics to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

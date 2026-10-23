
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.base import clone


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "processed" / "telco_churn_clean.csv"
RESULTS_PATH = ROOT_DIR / "data" / "processed" / "threshold_results.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_cv_model.joblib"

RANDOM_STATE = 42
TARGET = "Churn"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Load the model selected by cross-validation PR-AUC.
    model = joblib.load(MODEL_PATH)

    # Generate out-of-fold probabilities using training data only.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    print("\nTraining-only cross-validated PR-AUC:",
          round(average_precision_score(y_train, oof_probabilities), 4))

    rows = []

    for threshold in np.arange(0.20, 0.71, 0.02):
        predictions = (oof_probabilities >= threshold).astype(int)

        rows.append({
            "Threshold": round(float(threshold), 2),
            "Precision": precision_score(
                y_train, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_train, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_train, predictions, zero_division=0
            ),
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)

    print("\n===== THRESHOLD COMPARISON (TRAINING CV) =====")
    print(results.round(4).to_string(index=False))

    best_row = results.loc[results["F1"].idxmax()]

    print("\nThreshold with highest cross-validated F1:")
    print(f"Threshold: {best_row['Threshold']:.2f}")
    print(f"Precision: {best_row['Precision']:.4f}")
    print(f"Recall:    {best_row['Recall']:.4f}")
    print(f"F1-score:  {best_row['F1']:.4f}")

    print(f"\nSaved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()


from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run src/clean_data.py first."
        )

    # Load the cleaned dataset.
    df = pd.read_csv(DATA_PATH)

    # Separate target and features.
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # CustomerID is an identifier, not a useful model feature.
    X = df.drop(columns=["Churn", "customerID"])

    if y.isnull().any():
        raise ValueError("Unexpected values found in the Churn target.")

    # Identify numeric and categorical columns.
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Dataset shape:", df.shape)
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Churn rate:", round(y.mean() * 100, 2), "%")

    # Split before fitting any preprocessing steps.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Numeric preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Combine preprocessing for both feature types.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # Preprocessing and model are saved together in one pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # Evaluate on held-out test data.
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n===== BASELINE MODEL RESULTS =====")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(
        f"PR-AUC:    "
        f"{average_precision_score(y_test, y_prob):.4f}"
    )

    print("\n===== CONFUSION MATRIX =====")
    print("Rows = actual [No churn, Churn]")
    print("Columns = predicted [No churn, Churn]")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )

    # Save the complete fitted pipeline.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\n===== MODEL SAVED =====")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()

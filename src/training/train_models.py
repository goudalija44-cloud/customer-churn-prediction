
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"


def create_pipeline(model, numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run src/clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns=["Churn", "customerID"])

    if y.isnull().any():
        raise ValueError("Unexpected values in Churn target.")

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Use the same split for every model.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for name, estimator in models.items():
        print(f"\n{'=' * 55}")
        print(f"Training: {name}")

        pipeline = create_pipeline(
            estimator,
            numeric_features,
            categorical_features,
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
            "PR_AUC": average_precision_score(y_test, probabilities),
        }

        results.append(metrics)

        print(f"Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall:    {metrics['Recall']:.4f}")
        print(f"F1-score:  {metrics['F1']:.4f}")
        print(f"ROC-AUC:   {metrics['ROC_AUC']:.4f}")
        print(f"PR-AUC:    {metrics['PR_AUC']:.4f}")

        print("\nConfusion matrix (actual rows, predicted columns):")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        print("Saved:", MODEL_DIR / filename)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="PR_AUC", ascending=False
    )

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n===== MODEL COMPARISON =====")
    print(results_df.round(4).to_string(index=False))
    print("\nComparison saved to:", RESULTS_PATH)
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()

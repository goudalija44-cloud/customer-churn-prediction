
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "tuning_results.csv"


def build_pipeline(numeric_features, categorical_features, estimator):
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
        ("classifier", estimator),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
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

    # Keep the same holdout split as the baseline experiments.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    configurations = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "classifier__C": [0.1, 0.5, 1.0, 5.0, 10.0],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42,
            ),
            {
                "classifier__max_depth": [4, 6, 8, 12, None],
                "classifier__min_samples_leaf": [5, 10, 20],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [None, 10, 16],
                "classifier__min_samples_leaf": [3, 5, 10],
                "classifier__max_features": ["sqrt", 0.8],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [80, 100, 150],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [1, 2, 3],
            },
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    best_cv_score = -1.0
    best_model_name = None
    best_pipeline = None

    for name, (estimator, parameters) in configurations.items():
        print(f"\n{'=' * 55}")
        print(f"Tuning: {name}")

        pipeline = build_pipeline(
            numeric_features,
            categorical_features,
            estimator,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameters,
            n_iter=5,
            scoring="average_precision",
            cv=3,
            random_state=42,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_
        predictions = tuned_model.predict(X_test)
        probabilities = tuned_model.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "CV_PR_AUC": search.best_score_,
            "Test_Accuracy": accuracy_score(y_test, predictions),
            "Test_Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Test_Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "Test_F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "Test_ROC_AUC": roc_auc_score(y_test, probabilities),
            "Test_PR_AUC": average_precision_score(
                y_test, probabilities
            ),
            "Best_Parameters": str(search.best_params_),
        }
        results.append(row)

        print("Best CV PR-AUC:", round(search.best_score_, 4))
        print("Test PR-AUC:", round(row["Test_PR_AUC"], 4))
        print("Test recall:", round(row["Test_Recall"], 4))
        print("Best parameters:", search.best_params_)

        filename = name.lower().replace(" ", "_") + "_tuned.joblib"
        joblib.dump(tuned_model, MODEL_DIR / filename)

        if search.best_score_ > best_cv_score:
            best_cv_score = search.best_score_
            best_model_name = name
            best_pipeline = tuned_model

    results_df = pd.DataFrame(results).sort_values(
        "CV_PR_AUC", ascending=False
    )
    results_df.to_csv(RESULTS_PATH, index=False)

    # This is the best model by cross-validation, not necessarily
    # the final production choice.
    joblib.dump(best_pipeline, MODEL_DIR / "best_cv_model.joblib")

    print("\n===== TUNING COMPARISON =====")
    print(
        results_df[
            [
                "Model",
                "CV_PR_AUC",
                "Test_PR_AUC",
                "Test_Recall",
                "Test_Precision",
                "Test_F1",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nBest model by CV PR-AUC:", best_model_name)
    print("Results saved to:", RESULTS_PATH)
    print("Best CV model saved to:", MODEL_DIR / "best_cv_model.joblib")


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd



ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "best_cv_model.joblib"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "feature_importance.png"
CSV_PATH = ROOT_DIR / "data" / "processed" / "feature_importance.csv"

def main():
    model = joblib.load(MODEL_PATH)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances, 
    })

    importance_df = importance_df.sort_values(
        "Importance", ascending=False
    )

    importance_df.to_csv(CSV_PATH, index=False)

    top_features = importance_df.head(5).sort_values(
        "Importance", ascending=True
    )


    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Customer Feature")
    plt.title("Top 15 Features Influencing Churn Predictions")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n===== TOP 15 FEATURES ====")
    print(importance_df.head(15).to_string(index=False))

    print(f"\nSaved chart: {OUTPUT_PATH}")

    print(f"Saved feature data: {CSV_PATH}")


if __name__ == "__main__":
    main()
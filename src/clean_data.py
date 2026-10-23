
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "telco_churn_clean.csv"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {RAW_PATH}"
        )

    # Load the original data; do not overwrite it.
    df = pd.read_csv(RAW_PATH)

    print("Original dataset shape:", df.shape)

    # Remove accidental whitespace from column names and text values.
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Convert TotalCharges to numeric.
    # Blank strings become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    print("\nMissing values before cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Check duplicate customer IDs.
    duplicate_count = df["customerID"].duplicated().sum()
    print("\nDuplicate customer IDs:", duplicate_count)

    # Remove duplicate customer records if any exist.
    df = df.drop_duplicates(subset=["customerID"])

    # The original dataset has 11 blank TotalCharges values.
    # Inspect these records before deciding how to handle them.
    print("\nRows with missing TotalCharges:")
    print(
        df.loc[
            df["TotalCharges"].isnull(),
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
        ].to_string(index=False)
    )

    # These rows represent customers with zero tenure.
    # Their missing TotalCharges values are consistent with no
    # accumulated charges. Set them to 0 only when tenure is zero.
    zero_tenure_missing = (
        df["TotalCharges"].isnull() & df["tenure"].eq(0)
    )
    df.loc[zero_tenure_missing, "TotalCharges"] = 0

    # Stop rather than silently invent values for unexpected cases.
    if df["TotalCharges"].isnull().any():
        raise ValueError(
            "Unexpected missing TotalCharges values remain. "
            "Inspect these rows before continuing."
        )

    # Validate the churn target.
    print("\nChurn values:")
    print(df["Churn"].value_counts(dropna=False))

    if not set(df["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Unexpected values found in Churn.")

    if df["Churn"].isnull().any():
        raise ValueError("Missing values found in the Churn target.")

    # Save the cleaned data separately.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print("\n===== CLEANING SUMMARY =====")
    print("Cleaned dataset shape:", df.shape)
    print("Missing values remaining:", int(df.isnull().sum().sum()))
    print("Duplicate customer IDs remaining:",
          int(df["customerID"].duplicated().sum()))
    print("Saved cleaned dataset to:", CLEAN_PATH)


if __name__ == "__main__":
    main()

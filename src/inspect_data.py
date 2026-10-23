from pathlib import Path
import pandas as pd

# Locate the dataset relative to this python file
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "Telco-Customer-Churn.csv"
)


def main():
    print("Loading customer churn dataset...")

    if not DATA_PATH.exists():
        print(f"Dataset not found at: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    print("\n==== DATASET OVERVIEW ====")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n===== COLUMN NAMES =====")
    print(df.columns.tolist())

    print("\n==== FIRST 5 ROWS =====")
    print(df.head().to_string())

    print("\n==== DATA TYPES ====")
    print(df.dtypes.to_string())

    print("\n==== MISSING VALUES ====")
    print(df.isnull().sum().to_string())

    print("\n==== DUPLICATE CUSTOMER IDs ====")
    print(df["customerID"].duplicated().sum())

    print("\n==== CHURN DISTRIBUTION ====")
    print(df["Churn"].value_counts(dropna=False).to_string())

    print("\n==== TOTAL CHARGES CHECK ====")
    print("Data type:", df["TotalCharges"].dtype)
    print("Blank TotalCharges:", df["TotalCharges"].astype(str).str.strip().eq("").sum())

    print("\n Dataset inspection completed")


if __name__ == "__main__":
    main()
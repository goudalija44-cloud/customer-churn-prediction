
# from pathlib import Path

# import matplotlib.pyplot as plt
# import pandas as pd

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
# OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# def main():
#     df = pd.read_csv(DATA_PATH)

#     print("===== DATASET SUMMARY =====")
#     print(df.describe(include="all").to_string())

#     print("\n===== CHURN COUNTS =====")
#     print(df["Churn"].value_counts())

#     print("\n===== CHURN PERCENTAGES =====")
#     print(
#         (df["Churn"].value_counts(normalize=True) * 100)
#         .round(2)
#         .to_string()
#     )

#     print("\n===== CHURN BY CONTRACT =====")
#     print(
#         pd.crosstab(
#             df["Contract"],
#             df["Churn"],
#             normalize="index"
#         ).mul(100).round(2).to_string()
#     )

#     print("\n===== CHURN BY TENURE GROUP =====")
#     df["TenureGroup"] = pd.cut(
#         df["tenure"],
#         bins=[-1, 12, 24, 48, 72],
#         labels=["0-12", "13-24", "25-48", "49-72"]
#     )

#     print(
#         pd.crosstab(
#             df["TenureGroup"],
#             df["Churn"],
#             normalize="index"
#         ).mul(100).round(2).to_string()
#     )

#     # Plot churn distribution.
#     churn_counts = df["Churn"].value_counts().reindex(
#         ["No", "Yes"], fill_value=0
#     )

#     ax = churn_counts.plot(
#         kind="bar",
#         title="Customer Churn Distribution",
#         xlabel="Churn",
#         ylabel="Number of Customers",
#         rot=0
#     )

#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
#     plt.show()
#     plt.close()

#     # Plot churn rate by contract.
#     contract_churn = (
#         df.assign(ChurnFlag=df["Churn"].eq("Yes"))
#         .groupby("Contract", observed=False)["ChurnFlag"]
#         .mean()
#         .mul(100)
#         .sort_values(ascending=False)
#     )

#     ax = contract_churn.plot(
#         kind="bar",
#         title="Churn Rate by Contract Type",
#         xlabel="Contract",
#         ylabel="Churn Rate (%)",
#         rot=15
#     )

#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
#     plt.show()
#     plt.close()

#     print("\nEDA completed. Charts saved in:", OUTPUT_DIR)


# if __name__ == "__main__":
#     main()







from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET OVERVIEW =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n===== CHURN COUNTS =====")
    print(df["Churn"].value_counts())

    print("\n===== CHURN PERCENTAGES =====")
    print(
        (df["Churn"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

    print("\n===== CHURN RATE BY CONTRACT =====")
    contract_rates = (
        df.assign(ChurnFlag=df["Churn"].eq("Yes"))
        .groupby("Contract")["ChurnFlag"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )
    print(contract_rates.to_string())

    print("\n===== CHURN RATE BY TENURE GROUP =====")
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    )

    tenure_rates = (
        df.assign(ChurnFlag=df["Churn"].eq("Yes"))
        .groupby("TenureGroup", observed=True)["ChurnFlag"]
        .mean()
        .mul(100)
        .round(2)
    )
    print(tenure_rates.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Overall churn distribution
    churn_counts = df["Churn"].value_counts().reindex(
        ["No", "Yes"], fill_value=0
    )

    churn_counts.plot(
        kind="bar",
        title="Customer Churn Distribution",
        xlabel="Churn",
        ylabel="Number of Customers",
        rot=0,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
    plt.show()
    plt.close()

    # Chart 2: Churn rate by contract
    contract_rates.sort_values().plot(
        kind="barh",
        title="Churn Rate by Contract Type",
        xlabel="Churn Rate (%)",
        ylabel="Contract Type",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
    plt.show()
    plt.close()

    # Chart 3: Churn rate by tenure group
    tenure_rates.plot(
        kind="bar",
        title="Churn Rate by Customer Tenure",
        xlabel="Tenure (months)",
        ylabel="Churn Rate (%)",
        rot=0,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_tenure.png", dpi=150)
    plt.show()
    plt.close()

    print("\n===== EDA COMPLETE =====")
    print("Charts saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

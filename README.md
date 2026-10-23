# Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using customer demographics, service subscriptions, contract details, and billing information.

## Project Overview

Customer churn occurs when customers stop using a company's services. This project trains and evaluates classification models to estimate churn risk and provides predictions through a FastAPI backend and an interactive Streamlit dashboard.

## Features

- Data cleaning and exploratory data analysis
- Training and comparison of four classification models
- Hyperparameter tuning using RandomizedSearchCV
- Cross-validation and decision-threshold selection
- Evaluation using Precision, Recall, F1-score, ROC-AUC, and PR-AUC
- Feature-importance analysis
- REST API built with FastAPI
- Interactive Streamlit dashboard
- Automated API tests using pytest

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Backend:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Testing:** Pytest, HTTPX
- **Model Storage:** Joblib

## Dataset

IBM Telco Customer Churn dataset.

The dataset contains 7,043 customer records and includes customer information, service subscriptions, contract types, billing details, and the churn target.

## Model Performance

The selected Gradient Boosting model was evaluated using a held-out test set. The classification threshold was set to 0.34 to improve churn detection.

| Metric | Test Result |
|---|---:|
| Accuracy | 78.21% |
| Precision | 56.99% |
| Recall | 72.99% |
| F1-score | 64.01% |
| ROC-AUC | 84.55% |
| PR-AUC | 66.24% |

These results reflect the current evaluation split. Performance may vary on new customer data.

## Project Structure

```text
customer-churn-prediction/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── clean_data.py
│   └── eda.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Locally

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Start the FastAPI backend

```bash
python -m uvicorn api.main:app --reload
```

API base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 3. Start the Streamlit dashboard

Open a second terminal and run:

```bash
python -m streamlit run dashboard/app.py
```

### 4. Run automated tests

```bash
python -m pytest tests/test_api.py -v
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

## Deployment

Public deployment links will be added after deployment is complete.

- **Live dashboard:** To be added
- **Live API:** To be added
- **API documentation:** To be added

## Disclaimer

Predictions are estimates based on historical data and model behavior. They should support, not replace, informed business decisions. Feature importance indicates predictive relationships and does not establish causation.

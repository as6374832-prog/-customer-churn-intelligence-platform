# Customer Churn Intelligence Platform

An end-to-end machine learning project for analyzing telecom customer churn and estimating the probability that a customer may leave a service.

## Features

- Exploratory analysis of customer churn
- Visualizations of churn patterns
- Comparison of Logistic Regression, Random Forest, and Extra Trees
- Stratified cross-validation
- Customer churn probability predictions
- Downloadable prediction reports
- Saved machine learning model
- Automated tests with pytest

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Plotly · Matplotlib · Joblib · Pytest

## Project Structure

```text
customer-churn-project/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── models/
└── tests/
    └── test_model.py
```

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Run Automated Tests

```bash
python -m pytest tests/test_model.py -v
```

## Dataset

This project uses the Telco Customer Churn dataset. Place the CSV file inside the `data/` directory.

## Model Evaluation

The project compares Logistic Regression, Random Forest, and Extra Trees using ROC-AUC, precision, recall, and F1-score, with stratified cross-validation.

## Disclaimer

Churn probabilities are estimates based on historical data. They do not guarantee future customer behavior.

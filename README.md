# Customer Churn Intelligence Platform

A machine learning-powered customer churn prediction and analytics dashboard built with Python, Streamlit, and scikit-learn.

The platform helps explore customer churn patterns, compare classification models, and estimate churn risk for individual customers.

## Features

- **Interactive analytics dashboard** for exploring customer churn.
- **Customer churn prediction** using a trained machine learning pipeline.
- **Model comparison** across Logistic Regression, Random Forest, and Extra Trees.
- **Model evaluation** using ROC-AUC, precision, recall, and F1-score.
- **Stratified cross-validation** to evaluate model performance.
- **Custom prediction threshold** selected using training cross-validation predictions.
- **Prediction report export** to CSV.
- **Automated tests** for the saved model and prediction probability.

## Tech Stack

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- Plotly
- joblib
- pytest

## Project Structure

```text
customer-churn-project/
├── app.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── customer_churn_model.pkl
├── tests/
│   └── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

This project uses the Telco Customer Churn dataset. It contains customer demographics, subscribed services, contract details, billing information, and churn labels.

Place the CSV file in the `data/` directory using the filename expected by the application.

The dataset and trained model may be excluded from Git tracking depending on the repository's `.gitignore` settings. Ensure the required files are available when running the application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/as6374832-prog/-customer-churn-intelligence-platform.git
cd -customer-churn-intelligence-platform
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser.

## Model Evaluation

The project compares three classification algorithms:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline model |
| Random Forest | Ensemble model that captures nonlinear patterns |
| Extra Trees | Randomized tree-based ensemble |

Evaluation includes ROC-AUC, precision, recall, and F1-score. Stratified cross-validation is used for model comparison, and a separate held-out test set is used to report generalization performance.

The selected model is based on training cross-validation F1-score. The classification threshold is selected using training out-of-fold predictions.

Actual performance metrics should be reported from the latest successful application run rather than assumed or copied from earlier experiments.

## Run Automated Tests

With the virtual environment activated, run:

```bash
python -m pytest tests/test_model.py -v
```

The tests check that the saved model exists, can be loaded, and produces a valid churn probability.

## Prediction Output

For an individual customer, the dashboard displays an estimated churn probability and a predicted outcome. A prediction report can also be downloaded as a CSV file.

## Limitations

- Predictions are based on historical data and may not generalize to every business or customer population.
- Model performance depends on data quality and the selected classification threshold.
- A high estimated churn probability is not a guarantee that a customer will leave.
- This project is intended for educational and analytical demonstration, not as the sole basis for business decisions.

## Future Improvements

- Deploy the dashboard to a public hosting platform.
- Add model explainability using SHAP or permutation importance.
- Add data validation and monitoring.
- Improve the dashboard with additional customer segments and actionable retention insights.
- Add continuous integration to run tests automatically.

## Disclaimer

This project is an educational machine learning application. Its predictions are estimates, not guarantees of future customer behavior.


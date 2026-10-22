from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "customer_churn_model.pkl"


def test_saved_model_exists():
    assert MODEL_PATH.exists(), "Saved churn model was not found"


def test_saved_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    
def test_model_returns_valid_probability():
    model = joblib.load(MODEL_PATH)

    # Use one row from the original dataset as a realistic input
    import pandas as pd

    data_path = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["Churn", "customerID"]).head(1)

    probability = model.predict_proba(X)[0, 1]

    assert 0.0 <= probability <= 1.0
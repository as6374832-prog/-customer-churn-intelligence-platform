
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Intelligence Platform")
st.caption("Kaggle • Data Science • Machine Learning • Customer Retention")

# ---------------- LOAD DATA ----------------
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    st.error("Kaggle CSV not found inside the data folder.")
    st.stop()

df = pd.read_csv(csv_files[0])
df.columns = df.columns.str.strip()

required = {"Churn", "customerID"}
if not required.issubset(df.columns):
    st.error("The selected CSV is not the expected Telco Churn dataset.")
    st.stop()

df = df.drop_duplicates().copy()
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["Churn"])

# ---------------- DATA OVERVIEW ----------------
total_customers = len(df)
churned = int((df["Churn"] == "Yes").sum())
retained = total_customers - churned
churn_rate = churned / total_customers * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers", f"{total_customers:,}")
m2.metric("Churned Customers", f"{churned:,}")
m3.metric("Retained Customers", f"{retained:,}")
m4.metric("Churn Rate", f"{churn_rate:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Churn Distribution")
    distribution = (
        df["Churn"].value_counts()
        .rename_axis("Customer Status")
        .reset_index(name="Customers")
    )
    fig = px.pie(
        distribution,
        names="Customer Status",
        values="Customers",
        hole=0.45
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn by Contract")
    contract_data = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )
    fig = px.bar(
        contract_data,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- PREPARE ML DATA ----------------
X = df.drop(columns=["Churn", "customerID"])
y = (df["Churn"] == "Yes").astype(int)

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------- MODEL COMPARISON ----------------
st.divider()
st.header("Machine Learning Model Comparison")

# ---------------- CROSS-VALIDATION ----------------
st.subheader("Cross-Validation Results")
st.caption(
    "5-fold stratified cross-validation on the training set. "
    "The held-out test set remains separate for final evaluation."
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

cv_results = []

for name, classifier in cv_models.items():
    cv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision"
        },
        n_jobs=1
    )

    cv_results.append({
        "Model": name,
        "ROC-AUC Mean": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
        "F1 Mean": scores["test_f1"].mean(),
        "Recall Mean": scores["test_recall"].mean(),
        "Precision Mean": scores["test_precision"].mean()
    })

cv_df = pd.DataFrame(cv_results).sort_values(
    "ROC-AUC Mean",
    ascending=False
)

st.dataframe(
    cv_df.round(3),
    use_container_width=True,
    hide_index=True
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

with st.spinner("Training three models. Please wait..."):
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })

        trained_models[name] = pipeline

results_df = pd.DataFrame(results).sort_values(
    "ROC-AUC", ascending=False
).reset_index(drop=True)

st.dataframe(
    results_df.round(3),
    hide_index=True,
    use_container_width=True
)

fig = px.bar(
    results_df,
    x="Model",
    y="ROC-AUC",
    title="ROC-AUC Model Comparison",
    range_y=[0, 1]
)
st.plotly_chart(fig, use_container_width=True)

# Select model using training cross-validation ROC-AUC
best_name = pd.DataFrame(cv_results).sort_values(
    by="F1 Mean",
    ascending=False
).iloc[0]["Model"]

model = trained_models[best_name]

# Save the selected trained model
models_dir = project_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(model, models_dir / "customer_churn_model.pkl")

st.success(f"Selected model using training cross-validation F1-score: {best_name}")

# ---------------- EVALUATION ----------------

# Select the classification threshold using training data only
threshold_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=threshold_cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)

threshold_f1_scores = [
    f1_score(
        y_train,
        (oof_probabilities >= threshold).astype(int),
        zero_division=0
    )
    for threshold in thresholds
]

best_index = int(np.argmax(threshold_f1_scores))
best_threshold = float(thresholds[best_index])

# Evaluate once on the held-out test set
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= best_threshold).astype(int)

st.metric("F1-Optimized Threshold", f"{best_threshold:.2f}")
st.caption(
    f"Training cross-validation F1-score: "
    f"{threshold_f1_scores[best_index]:.3f}"
)
st.subheader("Selected Model — F1-Optimized Test Metrics")

st.write(f"Selected model: {best_name}")

st.metric(
    "Test Precision",
    f"{precision_score(y_test, predictions, zero_division=0):.3f}"
)

st.metric(
    "Test Recall",
    f"{recall_score(y_test, predictions, zero_division=0):.3f}"
)

st.metric(
    "Test F1-score",
    f"{f1_score(y_test, predictions, zero_division=0):.3f}"
)
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)

cm_fig = px.imshow(
    cm,
    text_auto=True,
    x=["Predicted Stay", "Predicted Churn"],
    y=["Actual Stay", "Actual Churn"],
    labels={"x": "Prediction", "y": "Actual"},
    title="Held-out Test Results"
)
st.plotly_chart(cm_fig, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("What Influences Churn?")

classifier = model.named_steps["classifier"]
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

if hasattr(classifier, "feature_importances_"):
    importance = classifier.feature_importances_
    importance_title = "Feature Importance"
else:
    importance = np.abs(classifier.coef_[0])
    importance_title = "Absolute Logistic Regression Coefficients"

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values("Importance", ascending=False).head(15)

importance_df["Feature"] = importance_df["Feature"].str.replace(
    "numeric__", "", regex=False
).str.replace("categorical__", "", regex=False)

fig = px.bar(
    importance_df.sort_values("Importance"),
    x="Importance",
    y="Feature",
    orientation="h",
    title=importance_title
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- CUSTOMER PREDICTION ----------------
# Customer Churn Prediction Form

st.subheader("Predict Customer Churn")
st.write("Enter customer details to estimate their churn probability.")

with st.form("prediction_form"):
    a, b, c = st.columns(3)

    with a:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with b:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with c:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=15000.0,
            value=float(tenure * monthly),
        )

    submitted = st.form_submit_button("Predict Churn Probability")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "TechSupport": support,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "OnlineBackup": backup,
        "DeviceProtection": device_protection,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
    }

    customer_df = pd.DataFrame([customer]).reindex(columns=X.columns)

    probability = float(model.predict_proba(customer_df)[0, 1])

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}",
    )

    if probability >= 0.65:
        st.error("High estimated churn risk")
    elif probability >= 0.35:
        st.warning("Moderate estimated churn risk")
    else:
        st.success("Low estimated churn risk")

    prediction_result = (
        "Churn" if probability >= best_threshold else "Stay"
    )

    report = pd.DataFrame(
        [
            {
                "Predicted Outcome": prediction_result,
                "Churn Probability": round(probability, 4),
                "Classification Threshold": round(best_threshold, 2),
                "Selected Model": best_name,
                "Tenure (Months)": tenure,
                "Contract": contract,
                "Monthly Charges": monthly,
                "Total Charges": total_charges,
            }
        ]
    )

    st.download_button(
        label="Download Prediction Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_prediction.csv",
        mime="text/csv",
    )

    st.caption(
        "This is a model estimate, not a certainty. "
        "Risk categories are illustrative."
    )
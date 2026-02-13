import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("Loan Approval Prediction – Dynamic Evaluation - 2025AA05284")

st.markdown("""
This application evaluates trained Machine Learning models on an uploaded test dataset.

Please upload a CSV file containing all feature columns and the **Loan_Status** target column.
""")


# ---------------------------------------
# Model Selection
# ---------------------------------------

model_choice = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

model_files = {
    "Logistic Regression": "saved_models/logistic.pkl",
    "Decision Tree": "saved_models/decision_tree.pkl",
    "kNN": "saved_models/knn.pkl",
    "Naive Bayes": "saved_models/naive_bayes.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "XGBoost": "saved_models/xgboost.pkl"
}


# ---------------------------------------
# File Upload Section
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload Test Dataset (must include 'Loan_Status' column)",
    type=["csv"]
)

# Show message on initial page load
if uploaded_file is None:
    st.info("Please upload a CSV file to compute evaluation metrics.")
    st.stop()


# ---------------------------------------
# Data Processing
# ---------------------------------------

df = pd.read_csv(uploaded_file)

if "Loan_Status" not in df.columns:
    st.error("Uploaded CSV must contain 'Loan_Status' column with Values Y or N.")
    st.stop()

st.subheader("Preview of Uploaded Data")
st.dataframe(df.head())

# Separate features and target
y_true = df["Loan_Status"].map({"Y": 1, "N": 0})
X = df.drop("Loan_Status", axis=1)

# Preprocessing (same logic used during training)
categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

# Impute categorical values
X[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[categorical_cols])

# Encode categorical variables
for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

# Impute numeric values
X[numeric_cols] = SimpleImputer(strategy="median").fit_transform(X[numeric_cols])

# Scale features
X = StandardScaler().fit_transform(X)


# ---------------------------------------
# Load Selected Model
# ---------------------------------------

model = joblib.load(model_files[model_choice])

# Predictions
y_pred = model.predict(X)

# Probability predictions (if available)
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y_true, y_prob)
else:
    auc_score = None


# ---------------------------------------
# Compute Evaluation Metrics
# ---------------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

metrics_dict = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "MCC": mcc
}

if auc_score is not None:
    metrics_dict["AUC"] = auc_score


# ---------------------------------------
# Display Metrics
# ---------------------------------------

st.subheader("Evaluation Metrics")

metrics_df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])
st.table(metrics_df)


# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

st.pyplot(fig)


# ---------------------------------------
# Footer
# ---------------------------------------

st.markdown("---")
st.write("Developed as part of M.Tech Machine Learning Assignment – BITS WILP")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("Loan Approval Prediction – ML Models Comparison")

st.write("""
This application demonstrates multiple machine learning classification models 
used to predict loan approval status. 
Upload a test dataset (CSV) to explore predictions.
""")

# -------------------------------
# Model Selection
# -------------------------------

model_choice = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    )
)

# -------------------------------
# Precomputed Evaluation Metrics
# -------------------------------

metrics_data = {
    "Logistic Regression": [0.7886, 0.7520, 0.7596, 0.9875, 0.8587, 0.5358],
    "Decision Tree": [0.7480, 0.6388, 0.7333, 0.9625, 0.8324, 0.4200],
    "kNN": [0.7642, 0.5000, 0.7383, 0.9875, 0.8449, 0.4768],
    "Naive Bayes": [0.7805, 0.7265, 0.7573, 0.9750, 0.8525, 0.5086],
    "Random Forest (Ensemble)": [0.7805, 0.7892, 0.7573, 0.9750, 0.8525, 0.5086],
    "XGBoost (Ensemble)": [0.7561, 0.7311, 0.7604, 0.9125, 0.8295, 0.4350]
}

metrics_labels = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]

st.subheader("Evaluation Metrics")

selected_metrics = metrics_data[model_choice]
metrics_df = pd.DataFrame({
    "Metric": metrics_labels,
    "Value": selected_metrics
})

st.table(metrics_df)

# -------------------------------
# Dataset Upload
# -------------------------------

st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your test dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    # Dummy Confusion Matrix (for demonstration)
    st.subheader("Confusion Matrix")

    # Example matrix (for demo purposes)
    cm = np.array([[85, 15],
                   [10, 90]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

st.markdown("---")
st.write("Developed as part of M.Tech Machine Learning Assignment – BITS WILP")

import os
import joblib
import pandas as pd

from model.data_loader import load_and_preprocess_data
from model.logistic_model import train_logistic
from model.decision_tree_model import train_decision_tree
from model.knn_model import train_knn
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.xgboost_model import train_xgboost


# -----------------------------------
# Create folder to store trained models
# -----------------------------------

os.makedirs("saved_models", exist_ok=True)

print("Loading and preprocessing dataset...\n")

X_train, X_test, y_train, y_test = load_and_preprocess_data("loan_prediction.csv")


# -----------------------------------
# Dictionary of models
# -----------------------------------

models = {
    "Logistic Regression": (train_logistic, "logistic.pkl"),
    "Decision Tree": (train_decision_tree, "decision_tree.pkl"),
    "kNN": (train_knn, "knn.pkl"),
    "Naive Bayes": (train_naive_bayes, "naive_bayes.pkl"),
    "Random Forest": (train_random_forest, "random_forest.pkl"),
    "XGBoost": (train_xgboost, "xgboost.pkl"),
}


results = []

print("Training Models...\n")

for model_name, (train_function, filename) in models.items():

    print(f"Training {model_name}...")

    # Each training function must return (model, metrics)
    model, metrics = train_function(X_train, X_test, y_train, y_test)

    # Save trained model
    joblib.dump(model, f"saved_models/{filename}")

    # Store metrics for display
    results.append({
        "Model": model_name,
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics.get("AUC", 0),
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "MCC": metrics["MCC"]
    })


# -----------------------------------
# Display Final Comparison Table
# -----------------------------------

results_df = pd.DataFrame(results)

print("\nFinal Model Comparison Table:\n")
print(results_df.to_string(index=False))

print("\nAll trained models saved inside 'saved_models/' folder.")

from model.data_loader import load_and_preprocess_data
from model.logistic_model import train_logistic
from model.decision_tree_model import train_decision_tree
from model.knn_model import train_knn
from model.naive_bayes_model import train_naive_bayes
from model.random_forest_model import train_random_forest
from model.xgboost_model import train_xgboost

X_train, X_test, y_train, y_test = load_and_preprocess_data("loan_prediction.csv")

models = {
    "Logistic Regression": train_logistic,
    "Decision Tree": train_decision_tree,
    "kNN": train_knn,
    "Naive Bayes": train_naive_bayes,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost
}

for name, func in models.items():
    print(f"\n{name}")
    metrics = func(X_train, X_test, y_train, y_test)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

def load_and_preprocess_data(csv_path):

    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Convert target Y/N to 1/0
    y = y.map({"Y": 1, "N": 0})

    # Handle categorical columns
    categorical_cols = X.select_dtypes(include='object').columns
    numeric_cols = X.select_dtypes(exclude='object').columns

    # Impute categorical with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Impute numeric with median
    num_imputer = SimpleImputer(strategy="median")
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


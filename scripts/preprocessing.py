import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

def preprocess_data(data):
    logging.info("Preprocessing data...")

    # Verify target column exists
    if 'SalePrice' not in data.columns:
        raise ValueError("Target column 'SalePrice' not found in data.")
    
    # Drop rows with missing target values
    data = data.dropna(subset=['SalePrice'])
    y = data['SalePrice']
    X = data.drop(columns=['SalePrice'])

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Impute missing values in numeric columns with median
    num_imputer = SimpleImputer(strategy='median')
    X_numeric = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

    # Impute missing values in categorical columns with mode
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_categorical = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)
        # One-hot encode categorical features
        X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)
        # Combine numeric and categorical features
        X_processed = pd.concat([X_numeric, X_categorical_encoded], axis=1)
    else:
        X_processed = X_numeric

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    return X_scaled, y
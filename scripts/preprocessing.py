
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    print("Preprocessing data...")
    
    # Drop rows with missing target
    data = data.dropna(subset=['SalePrice'])
    
    # Separate features and target variable
    X = data.drop(columns=['SalePrice'])
    y = data['SalePrice']
    
    # One-hot encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Standardize numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

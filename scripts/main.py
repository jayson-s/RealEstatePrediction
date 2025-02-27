import sys
import os
import logging
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from scripts.eda import perform_eda
from scripts.preprocessing import preprocess_data
from scripts.models import train_model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting the House Prices Prediction Project...")

    # Define the data path using os.path.join for cross-platform compatibility
    data_path = os.path.join("data", "house_prices.csv")
    
    # Check if the dataset exists
    if not os.path.exists(data_path):
        logging.error("Data file not found at: %s", data_path)
        sys.exit(1)
    
    logging.info("Loading dataset...")
    data = pd.read_csv(data_path)
    
    # Perform enhanced exploratory data analysis
    perform_eda(data)
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Train and evaluate multiple models
    train_model(X, y)

if __name__ == '__main__':
    main()
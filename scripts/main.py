
from scripts.eda import perform_eda
from scripts.preprocessing import preprocess_data
from scripts.models import train_model
import pandas as pd

def main():
    print("Starting the House Prices Prediction Project...")
    
    # Load the dataset
    data_path = "data/house_prices.csv"
    print("Loading dataset...")
    data = pd.read_csv(data_path)
    
    # Perform exploratory data analysis
    perform_eda(data)
    
    # Preprocess the data
    X, y = preprocess_data(data)
    
    # Train and evaluate models
    train_model(X, y)

if __name__ == '__main__':
    main()

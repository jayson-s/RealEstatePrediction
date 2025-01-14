
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    print("Performing EDA...")
    
    # Display dataset info
    print(data.info())
    
    # Check for missing values
    print(data.isnull().sum())
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("results/correlation_heatmap.png")
    plt.show()

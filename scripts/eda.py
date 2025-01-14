import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data):
    print("Performing EDA...")

    # Display dataset info
    print(data.info())

    # Check for missing values
    print(data.isnull().sum())

    # Compute correlation matrix for numeric columns only
    numeric_data = data.select_dtypes(include=["number"])  # Select only numeric columns
    correlation_matrix = numeric_data.corr()

    # Customize the heatmap for better readability
    plt.figure(figsize=(18,9))  # Adjust size for larger datasets
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=True,  # Display correlation values
        fmt=".2f",   # Format values to 2 decimal places
        cmap="coolwarm",  # Use a visually appealing colormap
        linewidths=0.65,  # Add gridlines between cells
        cbar_kws={"shrink": 0.75},  # Adjust colorbar size
        annot_kws={"size": 7}  # Set font size for annotations
    )
    
    # Improve labels
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,  # Rotate labels for better readability
        horizontalalignment="right",
        fontsize=10
    )
    heatmap.set_yticklabels(
        heatmap.get_yticklabels(),
        rotation=0,  # Keep vertical labels
        fontsize=10
    )

    # Add a title with better styling
    plt.title("Correlation Heatmap", fontsize=18, fontweight="bold", pad=20)
    
    # Save and show the plot
    plt.savefig("results/correlation_heatmap_improved.png", dpi=300, bbox_inches="tight")
    plt.show()
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

def perform_eda(data, results_dir="results"):
    logging.info("Performing EDA...")

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save dataset info to a file
    info_path = os.path.join(results_dir, "data_info.txt")
    with open(info_path, 'w') as f:
        data.info(buf=f)
    logging.info("Saved data info to %s", info_path)

    # Missing value analysis: missing values per column
    missing_values = data.isnull().sum()
    missing_values_path = os.path.join(results_dir, "missing_values.csv")
    missing_values.to_csv(missing_values_path)
    logging.info("Saved missing values analysis to %s", missing_values_path)

    # Correlation heatmap for numeric features
    numeric_data = data.select_dtypes(include=["number"])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(18, 9))
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.65,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 7}
    )
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)
    plt.title("Correlation Heatmap", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    heatmap_path = os.path.join(results_dir, "correlation_heatmap_improved.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved correlation heatmap to %s", heatmap_path)

    # Histograms for each numeric feature
    for col in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_data[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        hist_path = os.path.join(results_dir, f"hist_{col}.png")
        plt.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("Saved histogram for %s to %s", col, hist_path)

    # Boxplots for each numeric feature
    for col in numeric_data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=numeric_data[col])
        plt.title(f"Boxplot of {col}")
        boxplot_path = os.path.join(results_dir, f"boxplot_{col}.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("Saved boxplot for %s to %s", col, boxplot_path)

    # Pairplot for numeric features if there aren't too many columns
    if len(numeric_data.columns) <= 10:
        pairplot = sns.pairplot(numeric_data.dropna())
        pairplot_path = os.path.join(results_dir, "pairplot.png")
        pairplot.savefig(pairplot_path)
        plt.close()
        logging.info("Saved pairplot to %s", pairplot_path)
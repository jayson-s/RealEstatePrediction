import os
import logging
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X, y, results_dir="results"):
    logging.info("Training models...")

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models to evaluate
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    results_summary = {}

    for name, model in models.items():
        logging.info("Training %s...", name)
        # Hyperparameter tuning for RandomForest
        if name == "RandomForest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logging.info("Best parameters for RandomForest: %s", grid_search.best_params_)

        # Train the model
        model.fit(X_train, y_train)
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -np.mean(cv_scores)

        # Predict on test set and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results_summary[name] = {
            "CV_MSE": cv_mse,
            "Test_MSE": mse,
            "R2": r2
        }
        logging.info("%s - CV MSE: %.4f, Test MSE: %.4f, R2: %.4f", name, cv_mse, mse, r2)

        # Plot predictions vs. actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs. Predicted for {name}")
        plt.legend()
        plot_path = os.path.join(results_dir, f"{name}_predictions.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("Saved prediction plot for %s to %s", name, plot_path)

        # Save the trained model
        model_path = os.path.join(results_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        logging.info("Saved %s model to %s", name, model_path)

    # Save summary of results to CSV
    summary_df = pd.DataFrame(results_summary).T
    summary_path = os.path.join(results_dir, "model_results_summary.csv")
    summary_df.to_csv(summary_path)
    logging.info("Saved model results summary to %s", summary_path)
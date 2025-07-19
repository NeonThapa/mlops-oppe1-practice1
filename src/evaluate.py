# src/evaluate.py

import pandas as pd
from sklearn.metrics import r2_score
import joblib
import os

def evaluate_model():
    """
    Loads the saved model and the dataset, evaluates the model,
    and prints the R-squared score.
    """
    print("--- Starting Model Evaluation ---")

    # --- 1. Load Saved Model ---
    # The script is in src/, so we navigate one level up ('..') to find the model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'housing_model.joblib')
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file 'housing_model.joblib' not found. Please run 'train.py' first.")
        return

    # --- 2. Load Dataset ---
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'california_housing.csv')
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'california_housing.csv' not found.")
        return

    # --- 3. Prepare Data ---
    # This must be consistent with how the data was prepared for training
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    print("Data prepared for evaluation.")

    # --- 4. Make Predictions and Evaluate ---
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"Model R-squared Score: {r2:.4f}")
    print("--- Model Evaluation Finished ---")

# This block allows the script to be run directly from the command line
if __name__ == "__main__":
    evaluate_model()
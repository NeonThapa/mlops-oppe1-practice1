import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

def train_model():
    """
    Loads the California Housing dataset, trains a regression model,
    evaluates it, and saves it.
    """
    print("--- Starting Model Training (Regression) ---")

    # --- 1. Load Data ---
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'california_housing.csv')
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'california_housing.csv' not found. Please run 'create_dataset.py' first.")
        return

    # --- 2. Prepare Data ---
    # The last column 'MedHouseVal' is the target. All others are features.
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    print("Data prepared for training.")

    # --- 3. Train Model ---
    model = LinearRegression()
    model.fit(X, y)
    print("Model training complete.")

    # --- 4. Evaluate Model ---
    # For regression, we use metrics like R-squared (R2 score)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Model R-squared Score: {r2:.4f}")

    # --- 5. Save Model ---
    model_path = os.path.join(os.path.dirname(__file__), '..', 'housing_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    print("--- Model Training Finished ---")

if __name__ == "__main__":
    train_model()
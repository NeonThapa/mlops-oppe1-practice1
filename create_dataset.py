# create_dataset.py

from sklearn.datasets import fetch_california_housing
import pandas as pd

def create_csv():
    """
    Loads the California Housing dataset from scikit-learn
    and saves it as a CSV file in the current directory.
    """
    print("Fetching California Housing dataset...")
    # as_frame=True loads the data into a pandas DataFrame
    housing = fetch_california_housing(as_frame=True)
    
    # The data is in the 'frame' attribute
    df = housing.frame
    
    # The target column is already named 'MedHouseVal'
    
    file_path = "california_housing.csv"
    df.to_csv(file_path, index=False)
    
    print(f"Dataset saved successfully to: {file_path}")
    print("First 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    create_csv()
# tests/test_data_validation.py

import pandas as pd
import os

# Define the path to the dataset relative to the project root
DATASET_PATH = "california_housing.csv"

def test_csv_exists():
    """
    Tests if the california_housing.csv file exists in the project's root directory.
    """
    assert os.path.exists(DATASET_PATH), f"{DATASET_PATH} file not found!"

def test_data_columns():
    """
    Tests if the CSV has all the expected columns for the California Housing dataset.
    """
    df = pd.read_csv(DATASET_PATH)
    
    expected_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
        "AveOccup", "Latitude", "Longitude", "MedHouseVal"
    ]
    
    # Check 1: Are any of our expected columns missing from the file?
    missing_cols = [col for col in expected_columns if col not in df.columns]
    assert not missing_cols, f"CSV is missing expected columns: {missing_cols}"
    
    # Check 2: Does the file have extra columns that we don't expect?
    extra_cols = [col for col in df.columns if col not in expected_columns]
    assert not extra_cols, f"CSV has extra, unexpected columns: {extra_cols}"

def test_data_non_empty_and_shape():
    """
    Tests if the CSV is not empty and has a reasonable number of rows.
    """
    df = pd.read_csv(DATASET_PATH)
    
    assert not df.empty, "The CSV file is empty!"
    
    assert len(df) > 20000, f"The CSV file should have more than 20,000 rows, but found {len(df)}."
# tests/test_model.py

import subprocess
import os
import re # We'll use regular expressions to find the score in the script's output

def test_pipeline_and_model_performance():
    """
    Runs the full train-then-evaluate pipeline and checks the final R-squared score.
    """
    # --- 1. Run the training script to generate the model ---
    # This step ensures that 'housing_model.joblib' exists before we try to evaluate it.
    # We use 'check=True' to make the test fail immediately if the training script has an error.
    print("Running training script (src/train.py)...")
    subprocess.run(["python", "src/train.py"], check=True)

    # Assert that the model file was actually created.
    assert os.path.exists("housing_model.joblib"), "Training script did not create housing_model.joblib"

    # --- 2. Run the evaluation script ---
    # We capture the output of the evaluation script to check the metric.
    print("Running evaluation script (src/evaluate.py)...")
    result = subprocess.run(
        ["python", "src/evaluate.py"],
        capture_output=True,
        text=True, # Get output as a string
        check=True
    )

    # --- 3. Parse the R-squared score from the script's output ---
    output = result.stdout
    print(f"--- Output from evaluate.py ---\n{output}\n---------------------------------")

    # Use a regular expression to find the line with the score and extract the number.
    # This looks for the text "Model R-squared Score: " followed by a number (like 0.5939).
    match = re.search(r"Model R-squared Score: (\d+\.\d+)", output)
    
    # The test should fail if this line isn't found in the output.
    assert match is not None, "Could not find 'Model R-squared Score:' in the output of evaluate.py"
    
    # Convert the extracted number (which is a string) to a float.
    r2_score = float(match.group(1))
    print(f"Found R-squared score: {r2_score}")

    # --- 4. Assert that the score meets our quality threshold ---
    assert r2_score > 0.4, f"Model R-squared score {r2_score} is below the 0.4 threshold!"
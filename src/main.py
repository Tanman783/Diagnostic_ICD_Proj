# src/main.py
"""
ICD Code Project
Entry point for testing the environment and project setup
"""

"""def main():
    print("Welcome to the ICD Code Project!")
    # Example: simple test
    icd_codes = ["A00", "B20", "C34"]
    print("Sample ICD Codes:", icd_codes)

if __name__ == "__main__":
    main()"""

# src/main.py

# src/main.py
import numpy as np
from utils.evaluation import print_model_performance, plot_confusion_matrix

print("--- Testing Evaluation Utility ---")

# 1. Create fake data (simulating a model prediction)
# y_true: The actual answers (0 = No, 1 = Yes)
y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])

# y_pred: What our "dummy model" predicted
y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])

# y_prob: The probability confidence (needed for AUC)
y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.6, 0.8, 0.95, 0.3, 0.85])

# 2. Run the report
print_model_performance(y_true, y_pred, y_prob, model_name="Test Model")

# 3. Plot the matrix
# Note: This might open a popup window with the chart
print("\nGenerating plot...")
plot_confusion_matrix(y_true, y_pred)
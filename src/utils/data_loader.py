import pandas as pd
import os

def get_project_root():
    """Returns the root directory of the project."""
    # This gets the path of this file (data_loader.py)
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: utils -> src -> Diagnostic_ICD_Proj
    return os.path.dirname(os.path.dirname(current_path))

def load_data(subfolder, filename):
    """
    Loads a compressed CSV from the data directory.
    
    Args:
        subfolder (str): The folder inside 'data', e.g., 'mimic_cohort_NF_30_days'
        filename (str): The name of the file, e.g., 'mimic_cohort_NF_30_days.csv.gz'
    """
    root = get_project_root()
    # constructs path: D:/Projects/.../data/subfolder/filename
    file_path = os.path.join(root, 'data', subfolder, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")
    
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path, compression='gzip')
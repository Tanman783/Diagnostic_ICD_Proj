import pandas as pd
import numpy as np
import os
import pickle

def load_processed_data(cohort_path, icd_codes_path):
    """
    Loads the Master Data (Cohort Labels + ICD Codes).
    Args:
        cohort_path (str): Path to the cohort file (contains the Labels).
        icd_codes_path (str): Path to the codes file.
        
    Returns:
        df_target (pd.DataFrame): Cohort data with labels.
        df_codes (pd.DataFrame): ICD codes.
        valid_ids (list): List of unique hadm_ids from the Cohort file.
    """
    try:
        df_target = pd.read_csv(cohort_path, compression='gzip')
        df_codes = pd.read_csv(icd_codes_path, compression='gzip')
        
        if 'hadm_id' in df_target.columns:
            df_target['hadm_id'] = df_target['hadm_id'].astype(str)
        if 'hadm_id' in df_codes.columns:
            df_codes['hadm_id'] = df_codes['hadm_id'].astype(str)            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, []
    
    # We assume the cohort file dictates exactly which patients we are studying.
    valid_ids = df_target['hadm_id'].unique().tolist()
    
    print(f" Data Loaded. Total Patients in Cohort: {len(valid_ids)}")
    return df_target, df_codes, valid_ids

def get_fold_files(cv_folds_dir):
    """
    Returns a list of all .pkl fold files in the directory.
    """
    if not os.path.exists(cv_folds_dir):
        raise FileNotFoundError(f"Directory not found: {cv_folds_dir}")
        
    # Get all .pkl files
    files = [f for f in os.listdir(cv_folds_dir) if f.endswith('.pkl')]
    files.sort()
    
    full_paths = [os.path.join(cv_folds_dir, f) for f in files]
    return full_paths

def load_single_fold(pkl_path):
    """
    Loads a single .pkl file containing [Train, Val, Test].
    Extracts the 'hadm_id' (Column 1) from each array.
    """    
    with open(pkl_path, 'rb') as f:
        fold_data = pickle.load(f)
    
    def extract_hadm_ids(arr):
        if arr is None:
            return [] 
        arr = np.array(arr)
        # folds have: [Subject_ID, Hadm_ID, Label]
        # We MUST take Index 1 to get the Admission ID (200...)
        if len(arr.shape) > 1 and arr.shape[1] >= 2:
            ids = arr[:, 1]
        else:
            ids = arr
            
        return np.unique(ids).astype(str).tolist()

    train_ids = extract_hadm_ids(fold_data[0])
    val_ids   = extract_hadm_ids(fold_data[1])
    test_ids  = extract_hadm_ids(fold_data[2])
    
    return train_ids, val_ids, test_ids
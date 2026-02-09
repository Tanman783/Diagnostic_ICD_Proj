import pandas as pd
from src.utils.feature_selection import frequency


def create_binary_matrix(df_codes, code_col='icd_code'):
    """
    Converts (hadm_id, code) pairs into a binary matrix.
    Rows = Patients, Columns = Codes.
    """
    df_working = df_codes.copy()

    # Safety: Ensure IDs are strings
    df_working['hadm_id'] = df_working['hadm_id'].astype(str)

    # Pivot: Rows=hadm_id, Cols=Codes
    # unstack(fill_value=0) ensures we get 0 for missing codes
    matrix = df_working.groupby(['hadm_id', code_col]).size().unstack(fill_value=0)
    
    # # Binarize (0 or 1)
    matrix = (matrix > 0).astype(int)
    
    return matrix

def generate_feature_sets(df_codes, min_percentage=0.0):
    """
    Returns a dictionary: {'Full Codes': X, 'Groups': X, 'Combined': X}
    """
    # Apply Feature Selection
    if min_percentage > 0:
        df_filtered = frequency.filter_rare_codes(df_codes, min_percentage)
    else:
        df_filtered = df_codes.copy()
    
    # Generate 'Full Codes' Matrix
    X_full = create_binary_matrix(df_filtered, code_col='icd_code')

    # Generate 'Groups' Matrix (First 3 chars)
    df_groups = df_filtered.copy()
    df_groups['group'] = df_groups['icd_code'].astype(str).str[:3]
    X_groups = create_binary_matrix(df_groups, code_col='group')
    
    # Generate 'Combined' Matrix
    # We must align them to ensure the same patients exist in both
    common_index = X_full.index.intersection(X_groups.index)
    X_full = X_full.loc[common_index]
    X_groups = X_groups.loc[common_index]

    X_combined = pd.concat([X_full, X_groups], axis=1)
    
    # Return dictionary for the experiment loop
    return {
        "Full Codes": X_full,
        "Groups": X_groups,
        "Combined": X_combined
    }
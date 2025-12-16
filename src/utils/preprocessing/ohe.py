import pandas as pd

def create_binary_matrix(df_codes, code_col='icd_code'):
    """
    Helper: Converts a DataFrame of codes into a sparse binary matrix.
    """
    df_working = df_codes.copy()

    # Ensure we can group by hadm_id
    if df_working.index.name == 'hadm_id':
        df_working = df_working.reset_index()

    # Pivot: Rows=hadm_id, Cols=Codes
    matrix = df_working.groupby(['hadm_id', code_col]).size().unstack(fill_value=0)
    matrix = (matrix > 0).astype(int)
    
    return matrix

def generate_feature_sets(df_codes):
    """
    Returns a dictionary: {'Full Codes': X, 'Groups': X, 'Combined': X}
    """
    # Full Codes
    X_full = create_binary_matrix(df_codes, code_col='icd_code')

    # Groups (First 3 chars)
    # We work on a copy so we don't modify the original dataframe in memory
    df_groups = df_codes.copy()
    if df_groups.index.name == 'hadm_id':
        df_groups = df_groups.reset_index()
        
    df_groups['group'] = df_groups['icd_code'].str[:3]
    X_groups = create_binary_matrix(df_groups, code_col='group')
    
    # Align indices (ensure Groups has same patients as Full)
    X_groups = X_groups.reindex(X_full.index, fill_value=0)

    # Combined
    X_combined = pd.concat([X_full, X_groups], axis=1)

    return {
        "Full Codes": X_full,
        "Groups": X_groups,
        "Combined": X_combined
    }
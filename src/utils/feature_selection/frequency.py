import pandas as pd

def filter_rare_codes(df_codes, min_percentage=0.0):
    """
    Filters out ICD codes that appear in fewer than (min_percentage * total_patients).
    """
    if min_percentage <= 0:
        return df_codes

    # Calculate frequency based on unique patients per code
    total_patients = df_codes['hadm_id'].nunique()
    code_counts = df_codes.groupby('icd_code')['hadm_id'].nunique()
    
    # Define Threshold
    threshold = total_patients * min_percentage
    
    # Identify codes to keep
    keep_codes = code_counts[code_counts >= threshold].index
    
    # Filter DataFrame
    df_filtered = df_codes[df_codes['icd_code'].isin(keep_codes)].copy()
    
    print(f"  [Filter] Dropped codes < {min_percentage:.1%}. "
          f"Reduced from {len(code_counts)} to {len(keep_codes)} codes.")
    
    return df_filtered
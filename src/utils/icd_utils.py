import pandas as pd
import numpy as np

def clean_icd_code(code):
    """
    Standardizes an ICD code string to match the MIMIC cohort format.
    Operations:
    1. Converts to Uppercase (i10 -> I10)
    2. Removes dots (I10.9 -> I109)
    3. Strips whitespace (' I10 ' -> 'I10')
    """
    # Handle NaNs or empty values safely
    if pd.isna(code) or code == '':
        return ""
    # Force string, upper case, remove dots, strip spaces
    return str(code).upper().replace('.', '').strip()

def process_icd_column(df, column_name):
    """
    Applies the cleaner to an entire column in a DataFrame.
    Use this when loading a NEW embedding file before merging.
    """
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found. Available columns: {df.columns}")
        return df
    
    print(f"Normalizing ICD codes in column: '{column_name}'...")
    # Apply cleaning
    df[column_name] = df[column_name].apply(clean_icd_code)
    return df[column_name]

def check_match_rate(cohort_codes, embedding_codes):
    """
    Diagnostic tool: Checks how many of your cohort codes exist in the embeddings.
    Run this BEFORE merging to catch format issues.
    """
    cohort_set = set(cohort_codes.unique())
    embedding_set = set(embedding_codes.unique())
    
    matches = cohort_set.intersection(embedding_set)
    missing = cohort_set - embedding_set
    
    print(f"--- Merge Safety Check ---")
    print(f"Total unique codes in Cohort: {len(cohort_set)}")
    print(f"Total unique codes in Embeddings: {len(embedding_set)}")
    print(f"Matches found: {len(matches)}")
    print(f"Missing codes: {len(missing)}")
    print(f"Match Rate: {len(matches) / len(cohort_set):.2%}")
    
    return list(missing)
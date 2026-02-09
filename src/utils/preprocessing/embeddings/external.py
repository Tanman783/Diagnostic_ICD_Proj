import pandas as pd
import numpy as np
from src.utils.preprocessing.embeddings import base

def load_pretrained_embeddings(path):
    """
    Robust loader for Kane (CSV), Dryad (W2V/Txt), and Harvard (ClinVec) formats.
    Auto-detects separators and header styles.
    """
    path = str(path)
    print(f"Loading embeddings from: {path}...")
    
    try:
        df = None
        # STRATEGY 1: Try Standard CSV (Comma)
        if path.endswith('.csv') or path.endswith('.csv.gz'):
            try:
                df = pd.read_csv(path)
            except:
                pass # Failed? Fall through to Strategy 2
        
        # STRATEGY 2: Try Space/Tab separated
        if df is None:
            # 'header=None' because W2V files often don't have column names
            try:
                df = pd.read_csv(path, sep=r'\s+', header=None, skiprows=1) # 'skiprows=1' is a heuristic: often the first line is "Count Dimension" (e.g. 4000 100)
                df.columns = ['code'] + [f'v{i}' for i in range(df.shape[1]-1)] # Assign temporary column names: Code, v1, v2...
            except:
                df = pd.read_csv(path, sep=r'\s+', header=None)  # Last resort: Try reading without skipping (maybe no header line)
                df.columns = ['code'] + [f'v{i}' for i in range(df.shape[1]-1)]

        # --- NORMALIZE COLUMN NAMES ---
        # We need to find which column holds the ICD code string.
        code_col = None
        # Common names in these datasets:
        possible_names = ['code', 'icd_code', 'icd10', 'diagnosis', 'node_id', 'Code', 'key']
        
        for name in possible_names:
            if name in df.columns:
                code_col = name
                break
        
        # Fallback: If no known name found, assume the FIRST column is the code
        if code_col is None:
            code_col = df.columns[0]
            print(f"Warning: Could not identify code column by name. Using first column: '{code_col}'")

        # --- EXTRACT VECTORS ---
        # Identify vector columns (exclude metadata strings)
        meta_cols = [code_col, 'desc', 'description', 'label', 'definition']
        vector_cols = [c for c in df.columns if c not in meta_cols]
        
        embed_dict = {}
        for _, row in df.iterrows():
            # Clean Code: Remove dots (E11.9 -> E119), strip whitespace
            raw_code = str(row[code_col])
            clean_code = raw_code.replace(".", "").strip()
            
            # Extract Vector
            try:
                # Ensure we only grab numeric data
                vec = row[vector_cols].values.astype(np.float32)
                embed_dict[clean_code] = vec
            except ValueError:
                continue # Skip rows with parsing errors (headers etc)
            
        print(f" Loaded {len(embed_dict)} vectors with dimension {len(vector_cols)}.")
        return embed_dict, len(vector_cols)

    except Exception as e:
        print(f" Error loading embeddings: {e}")
        return None, 0

def create_feature_matrix(valid_ids, sequences_dict, embedding_path):
    """
    Orchestrates the loading and vectorization for Static Embeddings
    Returns a DataFrame ready for the Experiment.
    """
    # 1. Load Embeddings
    embed_dict, dim = load_pretrained_embeddings(embedding_path)
    
    if not embed_dict:
        return None

    # 2. Vectorize Patients
    X_matrix = base.vectorize_patients(
        hadm_ids=valid_ids, 
        sequences_dict=sequences_dict, 
        embedding_lookup=embed_dict, 
        vector_size=dim
    )
    
    # 3. Return as DataFrame (Indices aligned with valid_ids)
    return pd.DataFrame(X_matrix, index=valid_ids)
import pandas as pd
import numpy as np

def prepare_sequences(df_codes):
    """
    Converts DataFrame into a dictionary: {'hadm_id': ['code1', 'code2', ...]}
    Matches logic from 'get_admission_sequences'.
    """
    df_working = df_codes.copy()

    # Converts IDs to String (prevents Int vs Str mismatch)
    df_working['hadm_id'] = df_working['hadm_id'].astype(str)

    # Ensure codes are strings and remove trailing spaces ('Z5111  ' -> 'Z5111')
    if 'icd_code' in df_working.columns:
        df_working['icd_code'] = df_working['icd_code'].astype(str).str.strip()

    # Ensure hadm_id is a column
    if df_working.index.name == 'hadm_id':
        df_working = df_working.reset_index()
    
    # Sort by sequence number if available
    if 'seq_num' in df_working.columns:
        df_working = df_working.sort_values(['hadm_id', 'seq_num'])

    # Group by ID and collect codes into a list
    return df_working.groupby('hadm_id')['icd_code'].apply(list).to_dict()

def vectorize_patients(hadm_ids, sequences_dict, embedding_lookup, vector_size):
    """
    Maps a list of 'hadm_ids' to their average embedding vector.
    
    Args:
        embedding_lookup: 
            - If loading from external repo or pre trained embedding : Pass the dictionary loaded from external.py
            - If using Word2Vec: Pass `model.wv` (the KeyedVectors object)
    """
    matrix = []
    
    for pid in hadm_ids:
        pid = str(pid)
        codes = sequences_dict.get(pid, [])
        
        valid_vectors = []
        for code in codes:
            code = str(code)
            if code in embedding_lookup:
                valid_vectors.append(embedding_lookup[code])
        
        if valid_vectors:
            matrix.append(np.mean(valid_vectors, axis=0))
        else:
            matrix.append(np.zeros(vector_size))
            
    return np.array(matrix)
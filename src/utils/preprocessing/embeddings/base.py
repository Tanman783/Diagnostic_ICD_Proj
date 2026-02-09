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
        df_working['icd_code'] = df_working['icd_code'].astype(str).str.replace('.', '', regex=False).str.strip()

    # Ensure hadm_id is a column
    if df_working.index.name == 'hadm_id':
        df_working = df_working.reset_index()
    
    # Sort by sequence number if available
    if 'seq_num' in df_working.columns:
        df_working = df_working.sort_values(['hadm_id', 'seq_num'])

    # Group by ID and collect codes into a list
    return df_working.groupby('hadm_id')['icd_code'].apply(list).to_dict()


def vectorize_patients(hadm_ids, sequences_dict, embedding_lookup, vector_size, use_hierarchy_fallback=True):
    """
    Maps a list of 'hadm_ids' to their average embedding vector.
    
    Args:
        embedding_lookup: Dictionary or KeyedVectors object.
        use_hierarchy_fallback (bool): If True, tries to find parent codes for missing keys.
                                       (e.g., if 'E1191' is missing, try 'E119', then 'E11').
    """
    matrix = []
    
    # Helper to resolve vector with fallback logic
    def get_vector(code):
        # 1. Exact Match
        if code in embedding_lookup:
            return embedding_lookup[code]
            
        # 2. Hierarchy Fallback (Similarity Matching)
        if use_hierarchy_fallback:
            curr = code
            # Try stripping characters from the right until we find a match or hit length 3
            # E.g. E1191 -> E119 -> E11
            while len(curr) > 3: 
                curr = curr[:-1] 
                if curr in embedding_lookup:
                    return embedding_lookup[curr]
            
            # Final check for length 3 (Chapter/Category level)
            if len(curr) == 3 and curr in embedding_lookup:
                return embedding_lookup[curr]
        
        # 3. Fail -> None (will be filtered out)
        return None

    for pid in hadm_ids:
        pid = str(pid)
        codes = sequences_dict.get(pid, [])
        
        valid_vectors = []
        for code in codes:
            code = str(code)
            
            # Use the helper function instead of direct lookup
            vec = get_vector(code)
            
            # Only append if we found a valid vector (Exact or Parent)
            if vec is not None:
                valid_vectors.append(vec)
        
        if valid_vectors:
            matrix.append(np.mean(valid_vectors, axis=0))
        else:
            matrix.append(np.zeros(vector_size))
            
    return np.array(matrix)

def filter_and_vectorize(X_df, valid_codes_set, embedding_lookup, vector_size):
    """
    Filters the raw code lists in X_df and converts them to vectors.
    """
    # 1. Robustness: Handle NumPy inputs 
    if isinstance(X_df, np.ndarray):
        if X_df.ndim > 1 and X_df.shape[1] == 1:
             X_df = pd.DataFrame(X_df, columns=['codes'])
        else:
             X_df = pd.DataFrame(X_df, columns=['codes'])
             
    filtered_sequences = {}
    
    # 2. Iterate efficiently
    for hadm_id, row in X_df.iterrows():
        # Get the list of codes for this patient
        codes = row.iloc[0] 
        if not isinstance(codes, list):
            codes = []
            
        kept_codes = [c for c in codes if c in valid_codes_set]
        # Use str(hadm_id) as key.
        filtered_sequences[str(hadm_id)] = kept_codes    
        
    # 3. Vectorize 
    # Enable fallback so similarity matching is active
    return vectorize_patients(
        hadm_ids=X_df.index.tolist(),
        sequences_dict=filtered_sequences,
        embedding_lookup=embedding_lookup,
        vector_size=vector_size,
        use_hierarchy_fallback=True 
    )

def format_and_align_data(sequences_dict, df_target, valid_ids):
    """
    Converts the sequences dictionary into a DataFrame and aligns it with targets.
    """
    # 1. Convert Dict to DataFrame
    X_raw_df = pd.Series(sequences_dict, name='codes').to_frame()
    
    # 2. Ensure Index is String (to match valid_ids)
    X_raw_df.index = X_raw_df.index.astype(str)
    
    # 3. Align with valid_ids (Intersection)
    valid_ids_str = [str(pid) for pid in valid_ids]
    common_ids = [pid for pid in valid_ids_str if pid in X_raw_df.index]
    
    # Filter X
    X_raw_df = X_raw_df.loc[common_ids]
    
    # 4. Align Target
    df_target_aligned = df_target.copy()
    df_target_aligned['hadm_id'] = df_target_aligned['hadm_id'].astype(str)
    
    # Set index to 'hadm_id' and DO NOT reset it.
    df_target_aligned = df_target_aligned.set_index('hadm_id')
    
    # Align rows strictly with X
    df_target_aligned = df_target_aligned.loc[X_raw_df.index]
    
    return X_raw_df, df_target_aligned
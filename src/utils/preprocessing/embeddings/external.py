import pandas as pd
import numpy as np

def load_pretrained_embeddings(path):
    """
    Loads embeddings from a file path OR a URL.
    Args:
        path (str): The local file path or URL to the embeddings CSV.
    """
    print(f"Loading embeddings from: {path}...")
    
    try:
        compression = 'gzip' if path.endswith('.gz') else None
        df = pd.read_csv(path, compression=compression)
    except Exception as e:
        print(f" Error loading embeddings: {e}")
        return None

    # Identify vector columns dynamically
    meta_cols = ['code', 'desc', 'description']
    vector_cols = [c for c in df.columns if c not in meta_cols]
    
    embed_dict = {}
    
    for _, row in df.iterrows():
        # Strip dots to match mimic format
        clean_code = str(row['code']).replace(".", "").strip()
        embed_dict[clean_code] = row[vector_cols].values.astype(np.float32)
        
    print(f" Loaded {len(embed_dict)} vectors.")
    return embed_dict
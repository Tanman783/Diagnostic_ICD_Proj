import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 1. CONFIGURATION

current_file = Path(__file__).resolve()
project_root = current_file.parents[2] # src/utils -> src -> ROOT
data_dir = project_root / "data"
embedding_dir = data_dir / "embeddings"

# Input Files (Raw)
NODES_FILE = embedding_dir / "ClinGraph_nodes.csv"
VECTORS_FILE = embedding_dir / "ClinVec_icd10cm.csv"

# Output File (Cleaned)
OUTPUT_FILE = embedding_dir / "ClinVec_icd10cm_embeddings.csv"

# 2. PROCESSING LOGIC

def process_clinvec():
    print(f"Nodes File:   {NODES_FILE}")
    print(f"Vectors File: {VECTORS_FILE}")
    
    # Check existence
    if not NODES_FILE.exists() or not VECTORS_FILE.exists():
        print(f"[ERROR] Input files not found in {embedding_dir}")
        return
    # Step A: Load and Filter Nodes
    print("\n1. Loading Nodes Table...")

    try:
        df_nodes = pd.read_csv(NODES_FILE, sep='\t')
    except:
        df_nodes = pd.read_csv(NODES_FILE)

    print(f"   Raw Node Count: {len(df_nodes)}")

    # Filter for ICD-10 only: Check 'ntype' column OR 'node_id' string

    if 'ntype' in df_nodes.columns:
        mask = (df_nodes['ntype'] == 'ICD10CM') | \
               (df_nodes['node_id'].str.contains('icd10', case=False, na=False))
    else:
        mask = df_nodes['node_id'].str.contains('icd10', case=False, na=False)
    
    df_icd10 = df_nodes[mask].copy()
    print(f"   Filtered ICD-10 Count: {len(df_icd10)}")

    if len(df_icd10) == 0:
        print("[ERROR] No ICD-10 nodes found. Check the file format.")
        return

    # Create Mapping: node_index (int) -> clean_code (str)
    # Example: 10056 -> "R10.0:icd10cm" -> "R100"
    index_to_code = {}
    for _, row in df_icd10.iterrows():
        raw_id = str(row['node_id'])
        # Strip suffix like ':icd10cm' if it exists
        code_part = raw_id.split(':')[0]
        # Strip dots/whitespace to match MIMIC format (e.g. E11.9 -> E119)
        clean_code = code_part.replace(".", "").strip()
        
        index_to_code[row['node_index']] = clean_code

    print(f"   Created mapping for {len(index_to_code)} codes.")

    # Step B: Load and Map Vectors

    print("\n2. Loading Vectors Table...")
    # The vector file usually has an index column first
    df_vecs = pd.read_csv(VECTORS_FILE)
    
    # Identify index column
    index_col = df_vecs.columns[0]

    # Filter vectors to only those in our ICD-10 map
    df_vecs_filtered = df_vecs[df_vecs[index_col].isin(index_to_code.keys())].copy()
    print(f"   Matched {len(df_vecs_filtered)} vectors.")

    # Step C: Align and Save
    
    final_rows = []
    for _, row in df_vecs_filtered.iterrows():
        idx = row[index_col]
        
        if idx in index_to_code:
            code = index_to_code[idx]
            
            # Get vector values (exclude the index column)(convert to string)
            vector_values = row.values[1:]
            
            # Create a dictionary for the DataFrame. Structure: code, v0, v1, v2 ... 
            row_dict = {'code': code}
            for i, val in enumerate(vector_values):
                row_dict[f'v{i}'] = val
                
            final_rows.append(row_dict)

    df_final = pd.DataFrame(final_rows)
    
    # Save to CSV
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS. Saved {len(df_final)} aligned embeddings to:")
    print(f"{OUTPUT_FILE}")


# 3. EXECUTION

if __name__ == "__main__":
    process_clinvec()
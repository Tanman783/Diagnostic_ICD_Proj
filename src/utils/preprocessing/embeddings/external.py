import pandas as pd
import numpy as np
from src.utils import data_loader

def load_pretrained_embeddings(path):
    """
    Loads embeddings from a file path OR a URL.
    """
    print(f"Loading embeddings from: {path}...")
    try:
        compression = 'gzip' if path.endswith('.gz') else None
        df = pd.read_csv(path, compression=compression)
    except Exception as e:
        print(f" Error loading embeddings: {e}")
        return None

    meta_cols = ['code', 'desc', 'description']
    vector_cols = [c for c in df.columns if c not in meta_cols]
    
    embed_dict = {}
    for _, row in df.iterrows():
        clean_code = str(row['code']).replace(".", "").strip()
        embed_dict[clean_code] = row[vector_cols].values.astype(np.float32)
        
    print(f" Loaded {len(embed_dict)} vectors.")
    return embed_dict, len(vector_cols)

def run_cv_evaluation(X_df, df_target, fold_files, model_trainer_func, model_params={}, verbose=False):
    """
    Generic Cross-Validation Orchestrator for Static Embeddings .    
    Args:
        X_df: Pre-calculated DataFrame of patient vectors (Index=hadm_id).
        model_trainer_func: Callback function to train/evaluate a model.
    """
    fold_metrics = {'AUC': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    
    for i, fold_path in enumerate(fold_files):
        # Load Fold IDs
        train_ids, val_ids, test_ids = data_loader.load_single_fold(fold_path)
        
        # Split Data (Using the pre-calculated X_df)
        # Note: .reindex or .loc handles the splitting safely
        X_train = X_df.loc[train_ids].values
        X_val   = X_df.loc[val_ids].values
        X_test  = X_df.loc[test_ids].values
        
        y_train = df_target.set_index('hadm_id').loc[train_ids, 'label'].values
        y_val   = df_target.set_index('hadm_id').loc[val_ids, 'label'].values
        y_test  = df_target.set_index('hadm_id').loc[test_ids, 'label'].values
        
        # Delegate Training
        scores = model_trainer_func(
            X_train=X_train, y_train=y_train, 
            X_val=X_val, y_val=y_val, 
            X_test=X_test, y_test=y_test, 
            **model_params
        )
        
        if verbose:
            print(f"> Fold {i} AUC: {scores['AUC']:.4f}")
            
        for k, v in scores.items():
            if k in fold_metrics:
                fold_metrics[k].append(v)

    # 4. Aggregate
    avg_scores = {k: np.mean(v) for k, v in fold_metrics.items()}
    return avg_scores
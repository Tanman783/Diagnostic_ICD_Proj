import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from src.utils.preprocessing.embeddings import base
from src.utils import data_loader, evaluation
from src.utils.training import mlp

def train_word2vec(sequences_dict, vector_size=100, window=5, min_count=1, workers=4):
    """
    Trains a Word2Vec model on the provided sequences.
    """   
    sentences = list(sequences_dict.values())
    
    model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        sg=1, 
        workers=workers, 
        seed=42
    )
    return model

def get_fold_vectors(sequences_dict, train_ids, test_ids, val_ids=None, vector_size=100, window=5, min_count=1, workers=4):
    """
    Orchestrates W2V training and vectorization for a single fold to prevent leakage. 
    1. Filters sequences to use ONLY training data.
    2. Trains Word2Vec on that filtered data.
    3. Vectorizes all the sets using the new model.
    Args:
        val_ids (list, optional): If provided, returns X_val as well.
    
    Returns:
        If val_ids is None: (X_train, X_test) -> For Trees
        If val_ids is List: (X_train, X_val, X_test) -> For MLP/Early Stopping
    """
    # Leakage prevention: Filter sequences(We only show the Word2Vec model the histories of patients in the training set)
    train_sequences = {
        str(pid): sequences_dict[str(pid)] 
        for pid in train_ids 
        if str(pid) in sequences_dict
    }
    
    # Train Model
    model = train_word2vec(
        sequences_dict=train_sequences, vector_size=vector_size, window=window, min_count=min_count, workers=workers
    )

    # Vectorize Patients (Mapping IDs -> Vectors)
    # Vectorize Training Set
    X_train = base.vectorize_patients(
        hadm_ids=train_ids, sequences_dict=sequences_dict, embedding_lookup=model.wv, vector_size=vector_size
    )

    # Vectorize Test Set
    # (Safe to vectorize test patients now, because the MODEL didn't see them during training)
    X_test = base.vectorize_patients(
        hadm_ids=test_ids, sequences_dict=sequences_dict, embedding_lookup=model.wv, vector_size=vector_size
    )
    
    # Handle Validation Set ( for neural networks)
    if val_ids is not None:
        X_val = base.vectorize_patients(
            hadm_ids=val_ids, sequences_dict=sequences_dict, embedding_lookup=model.wv, vector_size=vector_size
        )
        return X_train, X_val, X_test
    
    return X_train, X_test

def run_cv_evaluation(sequences_dict, df_target, fold_files, vector_size, model_trainer_func, model_params={}, window=5, min_count=1, workers=1, verbose=False):
    """
    Generic Cross-Validation Orchestrator for Word2Vec.
    
    Args:
        model_trainer_func: A function that takes (X_train, y_train, X_val, y_val, X_test, y_test, **params)
                            and returns a dictionary of metrics {'AUC': 0.5, ...}.
        model_params: A dictionary of parameters to pass to the trainer (e.g., {'lr': 0.001} or {'n_estimators': 100}).
    """
    fold_metrics = {'AUC': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    
    for i, fold_path in enumerate(fold_files):
        # Load Fold IDs (Always get Train/Val/Test)
        train_ids, val_ids, test_ids = data_loader.load_single_fold(fold_path)
        
        # Generate Vectors (Dynamic training per fold to prevent leakage)
        # We always ask for Validation sets here subjected to be changed if to to use them or merge them with train set.
        X_train_raw, X_val_raw, X_test_raw = get_fold_vectors(
            sequences_dict=sequences_dict, train_ids=train_ids, test_ids=test_ids, val_ids=val_ids, vector_size=vector_size, window=window, min_count=min_count,
            workers=workers 
        )
        
        # Get Targets
        y_train = df_target.set_index('hadm_id').loc[train_ids, 'label'].values
        y_val   = df_target.set_index('hadm_id').loc[val_ids, 'label'].values
        y_test  = df_target.set_index('hadm_id').loc[test_ids, 'label'].values
        
        # We pass the raw data to the provided function. 
        # The function handles scaling, training, and evaluation.
        scores = model_trainer_func(
            X_train=X_train_raw, y_train=y_train, X_val=X_val_raw, y_val=y_val, X_test=X_test_raw, y_test=y_test, **model_params
        )
        
        # ONLY print if verbose is True
        if verbose:
            print(f"   > Fold {i} AUC: {scores['AUC']:.4f}")
            
        for k, v in scores.items():
            if k in fold_metrics:
                fold_metrics[k].append(v)

    # Aggregate
    avg_scores = {k: np.mean(v) for k, v in fold_metrics.items()}
    return avg_scores
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from src.utils.preprocessing.embeddings import base

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
        sg=1, # Skip-gram is usually better for medical codes
        workers=workers, 
        seed=42
    )
    return model

def generate_fold_embeddings(sequences_dict, train_ids, test_ids, val_ids=None, 
                           vector_size=100, window=5, min_count=1):
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
    # 1. Leakage prevention: Filter sequences(We only show the Word2Vec model the histories of patients in the training set)
    train_sequences = {
        str(pid): sequences_dict[str(pid)] 
        for pid in train_ids 
        if str(pid) in sequences_dict
    }
    
    # 2. Train Model
    model = train_word2vec(
        sequences_dict=train_sequences, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count
    )

    # 3. Vectorize Sets(Mapping IDs -> Vectors)
    # Note: We pass model.wv (KeyedVectors) as the lookup
    X_train = base.vectorize_patients(train_ids, sequences_dict, model.wv, vector_size)
    X_test = base.vectorize_patients(test_ids, sequences_dict, model.wv, vector_size)

    # Handle Validation Set ( for neural networks)
    if val_ids is not None:
        X_val = base.vectorize_patients(val_ids, sequences_dict, model.wv, vector_size)
        return X_train, X_val, X_test
    
    return X_train, X_test
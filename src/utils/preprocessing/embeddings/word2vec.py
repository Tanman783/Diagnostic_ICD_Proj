import numpy as np
import pandas as pd
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
        sg=1, 
        workers=workers, 
        seed=42
    )
    return model

def get_fold_vectors(sequences_dict, train_ids, test_ids, vector_size, window=5, min_count=1, workers=4):
    """
    Orchestrates W2V training and vectorization for a single fold to prevent leakage.
    
    1. Filters sequences to use ONLY training data.
    2. Trains Word2Vec on that filtered data.
    3. Vectorizes Train and Test patients using the new model.
    """
    
    # Leakage prevention: Filter sequences(We only show the Word2Vec model the histories of patients in the training set)
    train_sequences = {
        str(pid): sequences_dict[str(pid)] 
        for pid in train_ids 
        if str(pid) in sequences_dict
    }
    
    # Train Model (Dynamic Map)
    model = train_word2vec(
        sequences_dict=train_sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    # Vectorize Patients (Mapping IDs -> Vectors)
    # Vectorize Training Set
    X_train = base.vectorize_patients(
        hadm_ids=train_ids,
        sequences_dict=sequences_dict,
        embedding_lookup=model.wv,
        vector_size=vector_size
    )
    
    # Vectorize Test Set
    # (Safe to vectorize test patients now, because the MODEL didn't see them during training)
    X_test = base.vectorize_patients(
        hadm_ids=test_ids,
        sequences_dict=sequences_dict,
        embedding_lookup=model.wv,
        vector_size=vector_size
    )
    
    return X_train, X_test
import pandas as pd
import numpy as np
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from src.utils.preprocessing.embeddings import base
import warnings


warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# --- MODEL GETTERS ---

def get_xgboost_model():
    """Returns an untrained XGBoost classifier."""
    # enable_categorical=True is a good default, but our train_model 
    # function will handle the heavy lifting of data conversion.
    return XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="logloss", objective="binary:logistic", base_score=0.5,  n_jobs=-1, random_state=42,enable_categorical=True)

def get_rf_model():
    """Returns an untrained Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42)

def get_catboost_model():
    """Returns an untrained CatBoost classifier."""
    return CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6, verbose=0, allow_writing_files=False, random_state=42)

# HELPER FOR AUTOMATION

def get_model_instance(model_name):
    """
    Wrapper to get a fresh model instance by string name.
    Useful for looping through ['XGBoost', 'RandomForest', ...]
    """
    if model_name == "XGBoost":
        return get_xgboost_model()
    elif model_name == "RandomForest":
        return get_rf_model()
    elif model_name == "CatBoost":
        return get_catboost_model()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# TRAINING FUNCTION

def train_model(model_template, X_train, y_train):
    """
    Trains a model handling NumPy conversion and Class Imbalance automatically.
    """
    # Clone to ensure we don't accidentally train an already trained object
    model = clone(model_template)
    
    # NumPy Conversion (Avoids sklearn feature name warnings)
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        
    # Handle Class Imbalance for Boosting models (RF handles it via class_weight='balanced')
    if model.__class__.__name__ in ["XGBClassifier", "CatBoostClassifier"]:
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:
            ratio = float(np.sum(y_train == 0)) / n_pos
            model.set_params(scale_pos_weight=ratio)

    model.fit(X_train_np, y_train)
    return model

def tune_tree_hyperparameters(X_dev, y_dev, model_name, thresholds):
    """
    Accepts X_dev, y_dev and runs 3-Fold Inner CV.
    """
    best_avg_auc = -1
    best_thresh = thresholds[0]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    for t in thresholds:
        fold_scores = []
        
        # Inner Loop
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            # Split Data
            X_inner_train, X_inner_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_inner_train, y_inner_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
            
            # 1. Filter columns based on INNER TRAIN frequency
            min_count = int(t * len(X_inner_train))
            col_mask = (X_inner_train > 0).sum() >= min_count
            valid_cols = X_inner_train.columns[col_mask]
            
            if len(valid_cols) == 0:
                fold_scores.append(0.5)
                continue

            # 2. Train
            model_template = get_model_instance(model_name)
            model = train_model(model_template, X_inner_train[valid_cols], y_inner_train)
            
            # 3. Eval
            probs_val = get_probs(model, X_inner_val[valid_cols])
            try:
                score = roc_auc_score(y_inner_val, probs_val)
            except ValueError:
                score = 0.5
            fold_scores.append(score)
        
        # Calculate Average Score
        avg_score = np.mean(fold_scores)
        
        if avg_score > best_avg_auc:
            best_avg_auc = avg_score
            best_thresh = t
            
    return {
        'Selected_Threshold': best_thresh,
        'Val_AUC_Best': best_avg_auc
    }

def retrain_tree_model(X_combined, y_combined, best_params, model_name):
    """
    Standard retrainer for OHE data
    Callback for metrics.py:
    Applies the winning threshold to the combined data and retrains the final model.
    """
    threshold = best_params.get('Selected_Threshold', 0.0)
    
    # 1. Apply Logic: Filter columns based on Combined Data size
    min_count = int(threshold * len(X_combined))
    col_mask = (X_combined > 0).sum() >= min_count
    active_cols = X_combined.columns[col_mask]
    
    # 2. Retrain Final Model
    model_template = get_model_instance(model_name)
    final_model = train_model(model_template, X_combined[active_cols], y_combined)
    
    # Return BOTH the model AND the columns used
    return final_model, active_cols


# --- GENERIC EMBEDDING WRAPPER ---

class EmbeddingModelWrapper:
    """
    Wraps a Tree Model + Vectorizer into one object.
    When .predict() is called with Raw Data, it vectorizes
    """
    def __init__(self, model, embedding_dict, valid_codes, vector_size):
        self.model = model
        self.embedding_dict = embedding_dict
        self.valid_codes = valid_codes
        self.vector_size = vector_size
        # Expose classes_ so sklearn metrics know it's a classifier
        if hasattr(model, 'classes_'):
            self.classes_ = model.classes_

    def predict(self, X):
        # 1. Vectorize Raw Data (X is a DataFrame of lists)
        X_vec = base.filter_and_vectorize(X, self.valid_codes, self.embedding_dict, self.vector_size)
        # 2. Predict
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        # 1. Vectorize Raw Data
        X_vec = base.filter_and_vectorize(X, self.valid_codes, self.embedding_dict, self.vector_size)
        # 2. Predict Proba
        return self.model.predict_proba(X_vec)

def tune_static_embedding_model(X_dev, y_dev, model_name, thresholds, embedding_dict, vector_size):
    """
    Accepts X_dev, y_dev and runs 3-Fold Inner CV.
    """
    best_avg_auc = -1
    best_thresh = thresholds[0]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    for t in thresholds:
        fold_scores = []
        
        # Inner Loop
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            X_inner_train, X_inner_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_inner_train, y_inner_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            # 1. Filter Codes based on INNER TRAIN frequency
            all_codes = [c for sublist in X_inner_train.iloc[:,0] for c in sublist]
            code_counts = Counter(all_codes)
            min_count = int(len(X_inner_train) * t)
            valid_codes = {c for c, count in code_counts.items() if count >= min_count}
            
            # 2. Vectorize
            X_vec_train = base.filter_and_vectorize(X_inner_train, valid_codes, embedding_dict, vector_size)
            X_vec_val = base.filter_and_vectorize(X_inner_val, valid_codes, embedding_dict, vector_size)
            
            # 3. Train
            model_template = get_model_instance(model_name)
            model = train_model(model_template, X_vec_train, y_inner_train)
            
            # 4. Score
            probs_val = get_probs(model, X_vec_val)
            try:
                score = roc_auc_score(y_inner_val, probs_val)
            except ValueError:
                score = 0.5
            fold_scores.append(score)
            
        # Average Score
        avg_score = np.mean(fold_scores)
        
        if avg_score > best_avg_auc:
            best_avg_auc = avg_score
            best_thresh = t
            
    return {
        'Selected_Threshold': best_thresh,
        'Val_AUC_Best': best_avg_auc
    }

def retrain_static_embedding_model(X_combined, y_combined, best_params, model_name, embedding_dict, vector_size):
    """
    Generic Retrainer for Embedding Models.
    Returns a WRAPPED model that handles vectorization at inference time.
    """
    thresh = best_params.get('Selected_Threshold', 0.0)
    
    # 1. Calculate Valid Codes on Combined Data
    all_codes = [c for sublist in X_combined.iloc[:,0] for c in sublist]
    code_counts = Counter(all_codes)
    min_count = int(len(X_combined) * thresh)
    valid_codes = {c for c, count in code_counts.items() if count >= min_count}
    
    # 2. Vectorize Combined Data
    X_vec_combined = base.filter_and_vectorize(X_combined, valid_codes, embedding_dict, vector_size)
    
    # 3. Train Final Base Model
    model_template = get_model_instance(model_name)
    final_base_model = train_model(model_template, X_vec_combined, y_combined)
    
    # 4. Wrap it!
    final_wrapper = EmbeddingModelWrapper(final_base_model, embedding_dict, valid_codes, vector_size)
    
    # We return the RAW columns (which is just ['codes'])
    return final_wrapper, X_combined.columns


# --- WORD2VEC SPECIFIC TUNING & RETRAINING ---
from gensim.models import Word2Vec

def tune_w2v_model(X_dev, y_dev, model_name, thresholds, vector_size):
    """
    Aaccepts X_dev, y_dev and runs 3-Fold Inner CV.
    Trains W2V strictly on Inner Train to avoid leakage.
    """
    best_avg_auc = -1
    best_thresh = thresholds[0]
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def get_probs(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    for t in thresholds:
        fold_scores = []
        
        # Inner Loop
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            X_inner_train, X_inner_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_inner_train, y_inner_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            # A. Filter based on Inner Train
            all_codes = [c for sublist in X_inner_train.iloc[:,0] for c in sublist]
            code_counts = Counter(all_codes)
            min_count = int(len(X_inner_train) * t)
            valid_codes = {c for c, count in code_counts.items() if count >= min_count}
            
            # B. Train Word2Vec (On Inner Train Data Only)
            train_sentences = [
                [c for c in row if c in valid_codes] 
                for row in X_inner_train.iloc[:,0]
            ]
            
            w2v = Word2Vec(
                sentences=train_sentences,
                vector_size=vector_size,
                window=5, min_count=1, workers=4, epochs=10
            )
            w2v_kv = w2v.wv 
            
            # C. Vectorize Both
            X_vec_train = base.filter_and_vectorize(X_inner_train, valid_codes, w2v_kv, vector_size)
            X_vec_val = base.filter_and_vectorize(X_inner_val, valid_codes, w2v_kv, vector_size)
            
            # D. Train Tree
            model_template = get_model_instance(model_name)
            if model_name == "RandomForest":
                 model_template.set_params(max_depth=None)
                 
            model = train_model(model_template, X_vec_train, y_inner_train)
            
            # E. Evaluate
            probs_val = get_probs(model, X_vec_val)
            try:
                score = roc_auc_score(y_inner_val, probs_val)
            except ValueError:
                score = 0.5
            fold_scores.append(score)
            
        # Average Score
        avg_score = np.mean(fold_scores)
            
        if avg_score > best_avg_auc:
            best_avg_auc = avg_score
            best_thresh = t
            
    return {
        'Selected_Threshold': best_thresh,
        'Val_AUC_Best': best_avg_auc
    }

def retrain_w2v_model(X_combined, y_combined, best_params, model_name, vector_size):
    """
    Retrains the Final W2V + Tree Pipeline.
    """
    thresh = best_params.get('Selected_Threshold', 0.0)
    
    # 1. Filter
    all_codes = [c for sublist in X_combined.iloc[:,0] for c in sublist]
    code_counts = Counter(all_codes)
    min_count = int(len(X_combined) * thresh)
    valid_codes = {c for c, count in code_counts.items() if count >= min_count}
    
    # 2. Train Final Word2Vec
    train_sentences = [
        [c for c in row if c in valid_codes] 
        for row in X_combined.iloc[:,0]
    ]
    
    w2v = Word2Vec(
        sentences=train_sentences,
        vector_size=vector_size,
        window=5, min_count=1, workers=4, epochs=10
    )
    w2v_kv = w2v.wv
    
    # 3. Vectorize Combined
    X_vec_combined = base.filter_and_vectorize(X_combined, valid_codes, w2v_kv, vector_size)
    
    # 4. Train Final Tree
    model_template = get_model_instance(model_name)
    if model_name == "RandomForest":
         model_template.set_params(max_depth=None)

    final_base_model = train_model(model_template, X_vec_combined, y_combined)
    
    # 5. Wrap
    final_wrapper = EmbeddingModelWrapper(final_base_model, w2v_kv, valid_codes, vector_size)
    
    return final_wrapper, X_combined.columns
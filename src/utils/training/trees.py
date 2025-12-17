import pandas as pd
import numpy as np
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

def get_xgboost_model():
    """Returns an untrained XGBoost classifier."""
    # enable_categorical=True is a good default, but our train_model 
    # function will handle the heavy lifting of data conversion.
    return XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="logloss", n_jobs=-1, random_state=42,enable_categorical=True)

def get_rf_model():
    """Returns an untrained Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42)

def get_catboost_model():
    """Returns an untrained CatBoost classifier."""
    return CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=6, verbose=0, allow_writing_files=False, random_state=42)

def train_model(model_template, X_train, y_train):

    model = clone(model_template)
    
    # NumPy Conversion (Avoids sklearn feature name warnings)
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
        
    # Handling class Imbalance
   
    if model.__class__.__name__ in ["XGBClassifier", "CatBoostClassifier"]:
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:
            ratio = float(np.sum(y_train == 0)) / n_pos
            model.set_params(scale_pos_weight=ratio)

    model.fit(X_train_np, y_train)
    return model
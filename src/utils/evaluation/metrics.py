import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    f1_score, accuracy_score, precision_score, recall_score, average_precision_score
)
from src.utils import data_loader

def get_probs(model, X):
    """
    Universal helper to get probability scores from ANY model type.
    """
    if isinstance(X, pd.DataFrame): 
        X = X.values

    # Check for PyTorch Model
    if hasattr(model, 'parameters'): 
        model.eval()
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu') 

        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        else:
            X_tensor = X.to(device)
            
        with torch.no_grad():
            logits = model(X_tensor)
            if logits.shape[1] == 1: 
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            else: 
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            return probs
    else: 
        # Sklearn / XGBoost / CatBoost
        return model.predict_proba(X)[:, 1]

def get_preds(model, X, threshold=0.5):
    probs = get_probs(model, X)
    return (probs > threshold).astype(int)

def compute_metrics(model, X_test, y_test, threshold=0.5):
    """
    Calculates all metrics for a single fold.
    """
    probs = get_probs(model, X_test)
    preds = (probs > threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.5

    try:
        auprc = average_precision_score(y_test, probs)
    except ValueError:
        auprc = 0.0
        
    cm = confusion_matrix(y_test, preds)

    return {
        "AUC": auc, 
        "AUPRC": auprc, 
        "F1": f1_score(y_test, preds, zero_division=0), 
        "Accuracy": accuracy_score(y_test, preds), 
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0), 
        "Confusion_Matrix": cm
    }

def run_nested_cv_experiment(X, df_target, fold_files, tuning_callback, retraining_callback, model_name, feature_name="Standard", cohort_name="Unknown_Cohort"):
    """
    The Generic Orchestrator (Manager) for Nested Cross-Validation.
    Runs the pipeline on a SINGLE generic feature matrix X using a 5-Fold Outer Loop.
    this function combines Train + Val into a 'Development Set'. The tuning_callback is then responsible
    for performing 3-Fold Inner Cross-Validation on this Development Set.

    Args:
        X (pd.DataFrame): The feature matrix (Index MUST be hadm_id).
        df_target (pd.DataFrame): Labels (Must contain 'hadm_id' or be indexed by it).
        fold_files (list): Paths to .pkl files containing outer fold indices.
        tuning_callback (func): Function that accepts (X_dev, y_dev) and returns best_params.
        retraining_callback (func): Function that accepts (X_dev, y_dev, best_params) 
                                    and returns (fitted_model, active_features).
        model_name (str): Name for logging (e.g., 'XGBoost').
        feature_name (str): Label for this run (e.g., 'Full_Codes', 'W2V_Dim100').
        cohort_name (str): Label for the cohort (e.g., 'NF_30_Days').

    Returns:
        list: List of results dictionaries (one per outer fold).
    """
    results = []

    # Align Target Labels
    if 'hadm_id' in df_target.columns:
        y_global = df_target.set_index('hadm_id')['label']
    else:
        y_global = df_target['label']

    print(f"\nProcessing {model_name} on {feature_name} ({cohort_name})...")

    for fold_idx, fold_path in enumerate(fold_files):
        # 1. Load Indices (Outer Fold)
        train_ids, val_ids, test_ids = data_loader.load_single_fold(fold_path)
        
        # 2. Slice Data (Intersection with X ensures safety)
        valid_train = [pid for pid in train_ids if pid in X.index]
        valid_val   = [pid for pid in val_ids if pid in X.index]
        valid_test  = [pid for pid in test_ids if pid in X.index]

        X_train = X.loc[valid_train]
        y_train = y_global.loc[valid_train]
        
        X_val   = X.loc[valid_val]
        y_val   = y_global.loc[valid_val]
        
        X_test  = X.loc[valid_test]
        y_test  = y_global.loc[valid_test]

        # COMBINE FOR INNER CV
        # We merge Train and Val into a single "Development Set".
        # The tuning callback will handle the 3-fold split internally.
        X_dev = pd.concat([X_train, X_val])
        y_dev = pd.concat([y_train, y_val])
        
        # 3. PHASE A: TUNE
        # Pass combined data tuner
        best_params = tuning_callback(X_dev, y_dev)
        
        # 4. PHASE B: RETRAIN
        # Retrain on the same combined data using the best found parameters
        final_model, active_cols = retraining_callback(X_dev, y_dev, best_params)
        
        # 5. PHASE C: TEST
        # Evaluate on the held-out Outer Test Set
        if active_cols is not None:
            X_test_final = X_test[active_cols]
        else:
            X_test_final = X_test

        # We pass the selected threshold if it exists, else default to 0.5
        selected_threshold = best_params.get('Selected_Threshold', 0.5)
        scores = compute_metrics(final_model, X_test_final, y_test, threshold=selected_threshold)

        # Capture Final Model History
        # We check if the model has a 'history' attribute
        final_history = None
        if hasattr(final_model, 'history'):
            final_history = final_model.history
        elif hasattr(final_model, 'model') and hasattr(final_model.model, 'history'):
            final_history = final_model.model.history
        
        # If found, save -> plot.
        if final_history is not None:
            scores['History'] = final_history

        # 6. Record Results
        scores.update({
            'Cohort': cohort_name,
            'Feature_Type': feature_name, 
            'Model': model_name,
            'Fold': fold_idx,
            'Test_AUC': scores['AUC'],
            'Test_AUPRC': scores['AUPRC'],
            'Test_F1': scores['F1']
        })
        
        # Store the best parameters in the results for analysis
        scores.update(best_params)
        
        results.append(scores)
        print(f"  > Fold {fold_idx}: AUC={scores['AUC']:.4f} | Params={best_params}")

    return results
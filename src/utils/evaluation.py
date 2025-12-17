import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    f1_score, accuracy_score, precision_score, recall_score
)
import torch
import torch.nn as nn

def get_probs(model, X):
    """
    Universal helper to get probability scores from ANY model type.
    """
    # Convert DataFrame to Numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Check if it is a PyTorch Model
    if isinstance(model, nn.Module):
        model.eval()
        # Ensure input is a tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
            
        with torch.no_grad():
            # Forward pass -> squeeze to 1D array
            probs = model(X_tensor).squeeze().numpy()
            return probs

    # Otherwise, assume it is Sklearn/XGBoost/CatBoost
    else:
        # Return probability of class 1
        return model.predict_proba(X)[:, 1]

def get_preds(model, X, threshold=0.5):
    """
    Universal helper to get binary class labels (0/1).
    """
    probs = get_probs(model, X)
    return (probs > threshold).astype(int)

def compute_metrics(model, X_test, y_test, threshold=0.5):
    """
    Returns a dictionary of all key metrics.
    """
    probs = get_probs(model, X_test)
    preds = (probs > threshold).astype(int)
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = 0.0
        
    return {
        "AUC": auc,
        "F1": f1_score(y_test, preds),
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds)
    }

def cross_validate(model_template, X_data, y, fold_files, load_fold_func, train_func):
    """
    K-Fold Cross Validation.
    Args:
        model_template: The untrained model object.
        X_data: Full feature DataFrame.
        y: Full target Series.
        fold_files: List of paths to fold pickle files.
        load_fold_func: Function to load IDs from a fold file.
        train_func: Function to train the model.
    Returns:
        dict: Average metrics across folds.
    """
    # Initialize storage
    fold_metrics = {'AUC': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': []}
    
    for fold_path in fold_files:
        # Load Data
        train_ids, val_ids, test_ids = load_fold_func(fold_path)
        
        # Slice Data (Safety check for indices)
        train_mask = [pid for pid in (train_ids + val_ids) if pid in X_data.index]
        test_mask  = [pid for pid in test_ids if pid in X_data.index]
        
        X_train, y_train = X_data.loc[train_mask], y.loc[train_mask]
        X_test, y_test   = X_data.loc[test_mask], y.loc[test_mask]
        
        # Train (Delegated to the specific training function)
        model = train_func(model_template, X_train, y_train)
        
        # Score
        scores = compute_metrics(model, X_test, y_test)
        
        # Store
        for k, v in scores.items():
            fold_metrics[k].append(v)
            
    # Aggregate Results (Mean)
    return {k: np.mean(v) for k, v in fold_metrics.items()}

def plot_model_comparison(results_df, metric="AUC", title=None):
    """
    Plots a bar chart comparing models across different feature sets.
    """
    if title is None:
        title = f"Model Comparison ({metric})"
        
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=results_df,
        x="Features", y=metric, hue="Model", palette="viridis", edgecolor="black"
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel(f"Average {metric}", fontsize=11)
    plt.xlabel("Feature Representation", fontsize=11)
    
    # Dynamic Y-axis limit for better visualization
    min_score = results_df[metric].min()
    plt.ylim(max(0, min_score - 0.1), 1.0) 

    # Add value labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Algorithm')
    plt.tight_layout()
    plt.show()

def plot_cv_confusion_matrices(models_map, X_data, y, fold_files, load_fold_func, train_func):
    """
    Performs 5-Fold CV to collect predictions for EVERY patient in the dataset,
    then plots the aggregated Confusion Matrix for each model.
    """
    import math
    
    n_models = len(models_map)
    cols = min(n_models, 3)
    rows = math.ceil(n_models / cols)
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1: axes = [axes]
    else: axes = axes.flatten()
    
    fig.suptitle(f"Aggregated Confusion Matrices (All 5 Folds)", fontsize=16, y=1.02)
    
    # Loop through each model type
    for idx, (model_name, model_template) in enumerate(models_map.items()):
        # Storage for the aggregated results
        all_y_true = []
        all_y_pred = []
        
        # Run CV internally
        for fold_path in fold_files:
            # Load & Slice Data
            train_ids, val_ids, test_ids = load_fold_func(fold_path)
            
            # Filter IDs that exist in the feature set
            train_mask = [pid for pid in (train_ids + val_ids) if pid in X_data.index]
            test_mask  = [pid for pid in test_ids if pid in X_data.index]
            
            X_train, y_train = X_data.loc[train_mask], y.loc[train_mask]
            X_test, y_test   = X_data.loc[test_mask], y.loc[test_mask]
            
            # Train
            model = train_func(model_template, X_train, y_train)
            
            # Predict (Get Hard Labels 0/1)
            probs = get_probs(model, X_test)
            preds = (probs > 0.5).astype(int)
            
            # Collect
            all_y_true.extend(y_test)
            all_y_pred.extend(preds)

        # Plot aggregated matrix
        ax = axes[idx]
        cm = confusion_matrix(all_y_true, all_y_pred)
        
        # Calculate percentages for clearer medical interpretation
        # row_sums = cm.sum(axis=1)
        # cm_perc = cm / row_sums[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    # Cleanup layout
    for i in range(idx + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
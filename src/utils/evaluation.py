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
        X_tensor = torch.tensor(X, dtype=torch.float32)
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
    Returns a dictionary.
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



def plot_roc_curve(model, X_test, y_test, model_name="Model", ax=None):
    """
    Plots the ROC Curve.
    """
    probs = get_probs(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)

    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5) 
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """
    Plots the Confusion Matrix.
    """
    probs = get_probs(model, X_test)
    preds = (probs > 0.5).astype(int)
    
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()
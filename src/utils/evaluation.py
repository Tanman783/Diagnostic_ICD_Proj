import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

def get_performance_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standard binary classification metrics.
    
    Args:
        y_true: Actual labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for the positive class (for AUC)
        
    Returns:
        dict: A dictionary containing the metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        try:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['AUC-ROC'] = "N/A (Only one class present)"
            
    return metrics

def print_model_performance(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Prints a clean report of the model's performance.
    """
    print(f"--- Performance Report: {model_name} ---")
    metrics = get_performance_metrics(y_true, y_pred, y_prob)
    
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.4f}")
        else:
            print(f"{k:<15}: {v}")
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots a Confusion Matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()
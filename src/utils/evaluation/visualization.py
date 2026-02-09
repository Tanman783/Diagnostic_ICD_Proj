import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import math
import numpy as np
from sklearn.metrics import confusion_matrix

#  INTERNAL HELPER
def _save_plot(fig, output_path):
    """Standardizes saving and closing plots."""
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# PLOTTING FUNCTION

def plot_model_comparison(df_results, metric='Test_AUC', plots_dir=None, script_name=None):
    """
    Creates a bar chart comparing models based on the MEAN of a specific metric.
    Auto-detects whether to group by 'Dimension' (Embeddings) or 'Feature_Type' (Baseline).
    """
    if df_results.empty or metric not in df_results.columns:
        return

    plt.figure(figsize=(12, 6))
    
    # 1. Determine Logic: What variable are we comparing within the Model?
    # Priority: Dimension (MLP) > Feature_Type (Trees) > None
    if 'Dimension' in df_results.columns and df_results['Dimension'].nunique() > 1:
        hue_col = 'Dimension'
    elif 'Feature_Type' in df_results.columns and df_results['Feature_Type'].nunique() > 1:
        hue_col = 'Feature_Type'
    else:
        hue_col = None # Single color if no variation

    # 2. Plot
    sns.barplot(
        data=df_results, x='Model', y=metric, hue=hue_col, 
        palette='tab10', errorbar='sd',edgecolor='black', linewidth=1
    )
    
    plt.title(f'Model Comparison: {metric} (Mean ± Std)', fontsize=14)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Model', fontsize=12) 
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Move legend outside if it exists
    if hue_col:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title=hue_col)
    
    if script_name:
        plt.figtext(
            0.99, 0.01, 
            f"Source: {script_name}", 
            ha='right', fontsize=8, style='italic', color='gray'
        )

    if plots_dir:
        filename = f'comparison_{metric}.png'
        _save_plot(plt.gcf(), plots_dir / filename)


def plot_aggregated_confusion_matrices(df_results, plots_dir):
    """
    Plots ONE Combined Image containing side-by-side Confusion Matrices 
    for every unique configuration.
    """
    if 'Confusion_Matrix' not in df_results.columns:
        return

    cm_dir = plots_dir / 'confusion_matrices'
    cm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define Grouping
    group_cols = ['Model']
    if 'Dimension' in df_results.columns and df_results['Dimension'].nunique() > 1:
        group_cols.append('Dimension')
    elif 'Feature_Type' in df_results.columns and df_results['Feature_Type'].nunique() > 1:
        group_cols.append('Feature_Type')

    grouped = df_results.groupby(group_cols)
    
    # 2. Prepare Data
    plot_data = []
    for name, group in grouped:
        # Filter valid matrices
        matrices = group['Confusion_Matrix'].values
        valid_matrices = [m for m in matrices if isinstance(m, (np.ndarray, list))]
        if not valid_matrices: continue
        
        # Aggregate (Sum)
        agg_cm = np.sum(valid_matrices, axis=0)
        
        # Format Label
        if len(group_cols) > 1:
            # name is a tuple: (XGBoost, Full_Codes)
            label = f"{name[0]}\n({name[1]})"
        else:
            label = str(name)

        plot_data.append({'label': label, 'cm': agg_cm})

    if not plot_data: return

    # 3. Setup Grid (Max 3 columns wide)
    n_plots = len(plot_data)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # 4. Plot Each
    for i, data in enumerate(plot_data):
        ax = axes[i]
        cm = data['cm']
        
        # Calculate percentages
        cm_sum = np.sum(cm)
        cm_perc = cm / cm_sum if cm_sum > 0 else cm
        
        # Annotate: "Count\n(Perc%)"
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for r in range(nrows):
            for c in range(ncols):
                annot[r, c] = f"{cm[r, c]}\n({cm_perc[r, c]:.1%})"

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(data['label'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Aggregated Confusion Matrices (Summed across Folds)", fontsize=16)
    _save_plot(fig, cm_dir / "all_models_combined_cm.png")
    print(f" > Saved Combined CM Plot to: {cm_dir}")

def save_learning_curves(df_history, output_dir, experiment_name):
    """Generates 'Train vs Validation' plots. (MLP Only)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect column names
    t_col = next((c for c in ['train_loss', 'Train_Loss'] if c in df_history.columns), None)
    v_col = next((c for c in ['val_loss', 'Val_Loss'] if c in df_history.columns), None)
    
    if not t_col or not v_col: return

    # Grouping
    groups = df_history.groupby(['Dimension', 'Fold']) if 'Dimension' in df_history.columns else df_history.groupby('Fold')
    
    for name, group in groups:
        # Handle label safely
        label = f"Dim{name[0]}_Fold{name[1]}" if isinstance(name, tuple) else f"Fold{name}"
        
        fig = plt.figure(figsize=(8, 5))
        plt.plot(group['Epoch'], group[t_col], label='Train', color='blue')
        plt.plot(group['Epoch'], group[v_col], label='Val', color='orange', linestyle='--')
        plt.title(f"Learning Curve - {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _save_plot(fig, output_dir / f"lc_{label}_{experiment_name}.png")


# MASTER FUNCTIONS

def save_experiment_results(results, history_log, experiment_name, results_dir, plots_dir):
    """
    Saves raw metrics CSV and History (if provided).
    """
    results_dir, plots_dir = Path(results_dir), Path(plots_dir)
    
    # 1. Save Metrics CSV
    if results:
        df_results = pd.DataFrame(results) 
        keep_cols = [
            'Cohort', 'Model', 'Feature_Type', 'Fold', 'Dimension', 'Selected_Threshold', 'Best_Epoch', 'Val_AUC_Best','Test_AUC', 'Test_AUPRC', 'Test_F1', 'Test_Recall'
        ]
        final_cols = [c for c in keep_cols if c in df_results.columns]
        
        csv_path = results_dir / 'metrics.csv'
        df_results[final_cols].to_csv(csv_path, index=False)
        print(f" > Saved raw metrics to: {csv_path}")

    # 2. Save MLP History (If exists)
    if history_log:
        df_history = pd.concat(history_log, ignore_index=True)
        hist_path = results_dir / f'detailed_history_{experiment_name}.csv'
        df_history.to_csv(hist_path, index=False)
        
        # Generate Curves
        save_learning_curves(df_history, plots_dir / 'learning_curves', experiment_name)
        print(f" > Saved Learning Curves to: {plots_dir}")

def summarize_experiment(results, results_dir, plots_dir, group_by=['Model', 'Feature_Type'], script_name=None):
    """
    Calculates Mean/Std, saves Summary CSV, and generates Plots.
    """
    if not results: return

    df_results = pd.DataFrame(results)
    
    # Dynamic Grouping: Check if grouping cols actually exist
    valid_group_by = [c for c in group_by if c in df_results.columns]
    if not valid_group_by: valid_group_by = ['Model'] # Fallback

    # 1. Calculate Summary Stats
    metric_cols = [c for c in df_results.columns if c.startswith('Test_') and pd.api.types.is_numeric_dtype(df_results[c])]
    
    if metric_cols:
        summary = df_results.groupby(valid_group_by)[metric_cols].agg(['mean', 'std'])
        summary.columns = [f"{c[0]}_{c[1].capitalize()}" for c in summary.columns]
        summary = summary.reset_index()

        print("\n=== Model Performance Summary (Mean ± Std) ===")
        print(summary)
        summary.to_csv(results_dir / 'summary_metrics.csv', index=False)

        print(" > Generating Comparison Charts...")
        for metric in ['Test_AUC', 'Test_AUPRC', 'Test_F1']:
            if metric in df_results.columns:
                plot_model_comparison(df_results, metric=metric, plots_dir=plots_dir, script_name=script_name)
            
    # 2. Generate Confusion Matrix Plot
    plot_aggregated_confusion_matrices(df_results, plots_dir)




def save_aggregated_validation_curves(history_log, output_dir):
    """
    Plots validation loss of ALL folds on a single chart for each Dimension.
    Useful for checking stability (variance) across folds.
    """
    if not history_log: return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine list of DFs into one huge DF
    df_all = pd.concat(history_log, ignore_index=True)
    
    # Detect validation column
    val_col = next((c for c in ['val_loss', 'Val_Loss'] if c in df_all.columns), None)
    if not val_col: return

    # Group by Dimension (we want one plot per Dimension)
    if 'Dimension' in df_all.columns:
        groups = df_all.groupby('Dimension')
    else:
        groups = [('Default', df_all)]

    for dim_name, group in groups:
        plt.figure(figsize=(10, 6))
        
        # Plot each fold individually
        for fold in sorted(group['Fold'].unique()):
            fold_data = group[group['Fold'] == fold]
            plt.plot(fold_data['Epoch'], fold_data[val_col], alpha=0.3, label=f'Fold {fold}')
        
        # Calculate and plot the Average across folds
        # We group by Epoch to get the mean loss at step 1, step 2, etc.
        avg_data = group.groupby('Epoch')[val_col].mean()
        plt.plot(avg_data.index, avg_data.values, color='black', linestyle='--', linewidth=2, label='Average')

        plt.title(f"Stability Check: Validation Loss (Dim {dim_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        filename = f"stability_val_loss_Dim{dim_name}.png"
        _save_plot(plt.gcf(), output_dir / filename)
        print(f" > Saved Stability Plot: {output_dir / filename}")
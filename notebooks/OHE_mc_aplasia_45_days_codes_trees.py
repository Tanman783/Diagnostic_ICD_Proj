import sys
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial

# 1. SETUP PATHS & IMPORTS

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Custom Modules
from src.utils import data_loader
from src.utils.evaluation import metrics, visualization
from src.utils.preprocessing import ohe
from src.utils.training import trees
import warnings
# Filter out the specific sklearn warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 2. CONFIGURATION


EXPERIMENT_NAME = "OHE_mc_aplasia_45_days_codes_trees"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes"
THRESHOLDS = [0.0, 0.001, 0.005, 0.01] # 0%, 0.1%, 0.5%, 1%

MODELS_TO_RUN = ["XGBoost", "RandomForest", "CatBoost"]

# Paths
DATA_DIR = project_root / 'data' / COHORT_NAME
FOLDS_DIR = DATA_DIR / 'cv_folds'

# Output Paths
RESULTS_DIR = project_root / 'results' / 'experiments' / CODE_TYPE / EXPERIMENT_NAME
PLOTS_DIR = project_root / 'results' / 'plots' / CODE_TYPE / EXPERIMENT_NAME

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. LOAD DATA
print(f"Loading data from: {DATA_DIR}")

cohort_path = DATA_DIR / f"{COHORT_NAME}.csv.gz"
codes_path = DATA_DIR / f"{COHORT_NAME}_icd_codes.csv.gz" 

# Load raw frames
df_target, df_codes, _ = data_loader.load_processed_data(cohort_path, codes_path)
fold_files = data_loader.get_fold_files(FOLDS_DIR)

print(f"Data Loaded. Found {len(fold_files)} folds.")


# 4. FEATURE ENGINEERING (OHE)
print("Generating One-Hot Encoded Features...")

# We use the ohe module to generate sparse matrices (Full & Groups)
# Passing min_percentage=0.0 because filtering happens inside trees.py logic
feature_sets = ohe.generate_feature_sets(
    df_codes=df_codes, 
    min_percentage=0.0 
)


# 5. EXPERIMENT LOOP


all_results = []

for feat_name, X in feature_sets.items():
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {feat_name}")
    print(f"{'='*50}")
    
    for model_name in MODELS_TO_RUN:
        
        # 1. Tuning
        tuning_strategy = partial(
            trees.tune_tree_hyperparameters,
            model_name=model_name,
            thresholds=THRESHOLDS
        )
        
        # 2. Retraining
        retraining_strategy = partial(
            trees.retrain_tree_model,
            model_name=model_name
        )
        
        # 3. Call Generic Orchestrator
        fold_results = metrics.run_nested_cv_experiment(
            X=X, df_target=df_target, fold_files=fold_files, tuning_callback=tuning_strategy, retraining_callback=retraining_strategy,
            model_name=model_name, feature_name=feat_name, cohort_name=COHORT_NAME
        )
        
        all_results.extend(fold_results)

# 6. SAVE & VISUALIZE

print(f"\nSaving results to: {RESULTS_DIR}")

# 1. Save Raw Metrics
visualization.save_experiment_results(
    all_results, 
    None, # No learning curves for trees
    f"{EXPERIMENT_NAME}_metrics", 
    RESULTS_DIR, 
    PLOTS_DIR
)

# 2. Generate Summary & Plots
# Groups by [Model, Feature_Type] to compare XGBoost-Full vs XGBoost-Groups
visualization.summarize_experiment( all_results, RESULTS_DIR, PLOTS_DIR, group_by=['Model', 'Feature_Type'], script_name=EXPERIMENT_NAME)

print("Experiment Complete.")
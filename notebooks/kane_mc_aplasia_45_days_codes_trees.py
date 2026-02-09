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

from src.utils import data_loader
from src.utils.evaluation import metrics, visualization
from src.utils.preprocessing.embeddings import base, external 
from src.utils.training import trees

# 2. CONFIGURATION

EXPERIMENT_NAME = "kane_mc_aplasia_45_days_codes_trees"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes" 

# Dynamic Feature Selection Thresholds (0%, 0.1%, 0.5%, 1%)
THRESHOLDS = [0.0, 0.001, 0.005, 0.01]
MODELS_TO_RUN = ["XGBoost", "RandomForest", "CatBoost"]

# Local Embedding Files
EMBEDDING_FILES = {
    10: "icd-10-cm-2022-0010.csv.gz",
    50: "icd-10-cm-2022-0050.csv.gz",
    100: "icd-10-cm-2022-0100.csv.gz",
    1000: "icd-10-cm-2022-1000.csv.gz"
}

# Paths
DATA_DIR = project_root / 'data' / COHORT_NAME
FOLDS_DIR = DATA_DIR / 'cv_folds'
EMBEDDING_DIR = project_root / 'data' / 'embeddings'

RESULTS_DIR = project_root / 'results' / 'experiments' / CODE_TYPE / EXPERIMENT_NAME
PLOTS_DIR = project_root / 'results' / 'plots' / CODE_TYPE / EXPERIMENT_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. LOAD & PREPARE DATA

print(f"Loading data from: {DATA_DIR}")
cohort_path = DATA_DIR / f"{COHORT_NAME}.csv.gz"
codes_path = DATA_DIR / f"{COHORT_NAME}_{CODE_TYPE}.csv.gz"

df_target, df_codes, valid_ids = data_loader.load_processed_data(cohort_path, codes_path)
fold_files = data_loader.get_fold_files(FOLDS_DIR)

# Prepare Sequences
print("Converting patient history to sequences...")
sequences_dict = base.prepare_sequences(df_codes)

# Format & Align
X_raw_df, df_target = base.format_and_align_data(sequences_dict, df_target, valid_ids)

print(f"Aligned Data. Patients: {len(X_raw_df)}. Folds: {len(fold_files)}")

# 4. EXPERIMENT LOOP

all_results = []

for dim, filename in EMBEDDING_FILES.items():

    print(f"Processing Dimension {dim}") 
    # A. Load Embeddings
    local_path = EMBEDDING_DIR / filename
    if not local_path.exists():
        print(f"[ERROR] Embedding file not found: {local_path}")
        continue
    
    embedding_dict, loaded_dim = external.load_pretrained_embeddings(local_path)
    if not embedding_dict: continue
    
    # B. Run Models
    for model_name in MODELS_TO_RUN:
        
        # 1. Tuning Strategy
        tuning_strategy = partial(
            trees.tune_static_embedding_model,
            model_name=model_name,
            thresholds=THRESHOLDS,
            embedding_dict=embedding_dict,
            vector_size=loaded_dim
        )
        
        # 2. Retraining Strategy (Using Wrapper)
        retraining_strategy = partial(
            trees.retrain_static_embedding_model,
            model_name=model_name,
            embedding_dict=embedding_dict,
            vector_size=loaded_dim
        )
        
        # 3. Run Manager
        fold_results = metrics.run_nested_cv_experiment(
            X=X_raw_df, 
            df_target=df_target,
            fold_files=fold_files,
            tuning_callback=tuning_strategy,
            retraining_callback=retraining_strategy,
            model_name=model_name,
            feature_name=f"Kane_{dim}", 
            cohort_name=COHORT_NAME
        )
        
        # Inject Meta-data
        for res in fold_results:
            res['Dimension'] = dim
            res['Feature_Type'] = "Kane"
            
        all_results.extend(fold_results)

# 5. SAVE & VISUALIZE
print(f"\nSaving results to: {RESULTS_DIR}")
visualization.save_experiment_results(all_results, None, f"{EXPERIMENT_NAME}_metrics", RESULTS_DIR, PLOTS_DIR)
visualization.summarize_experiment( all_results, RESULTS_DIR, PLOTS_DIR, group_by=['Model', 'Feature_Type', 'Dimension'], script_name=EXPERIMENT_NAME)

print("Experiment Complete.")
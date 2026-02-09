import sys
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial

# SETUP PATHS
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import data_loader
from src.utils.evaluation import metrics, visualization
from src.utils.preprocessing.embeddings import base, external
from src.utils.training import mlp

# CONFIGURATION
EXPERIMENT_NAME = "kane_mc_aplasia_45_days_codes_mlp2layer"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes" 
NUM_LAYERS = 2

# Dynamic Feature Selection Thresholds
THRESHOLDS = [0.0, 0.001, 0.005, 0.01]
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

# LOAD DATA
print(f"Loading data from: {DATA_DIR}")
cohort_path = DATA_DIR / f"{COHORT_NAME}.csv.gz"
codes_path = DATA_DIR / f"{COHORT_NAME}_{CODE_TYPE}.csv.gz"

df_target, df_codes, valid_ids = data_loader.load_processed_data(cohort_path, codes_path)
fold_files = data_loader.get_fold_files(FOLDS_DIR)

# Prepare Sequences
print("Converting patient history to sequences...")
sequences_dict = base.prepare_sequences(df_codes)
X_raw_df, df_target = base.format_and_align_data(sequences_dict, df_target, valid_ids)

print(f"Aligned Data. Patients: {len(X_raw_df)}. Folds: {len(fold_files)}")

# EXPERIMENT LOOP
all_results = []

for dim, filename in EMBEDDING_FILES.items():
    print(f"\n{'='*50}")
    print(f"Processing Dimension {dim} (MLP {NUM_LAYERS}-Layer)")
    print(f"{'='*50}")
    
    # A. Load Embeddings
    local_path = EMBEDDING_DIR / filename
    if not local_path.exists():
        print(f"[ERROR] Embedding file not found: {local_path}")
        continue
    
    embedding_dict, loaded_dim = external.load_pretrained_embeddings(local_path)
    if not embedding_dict: continue
    
    # B. Define Strategy
    tuning_strategy = partial(
        mlp.tune_static_generic_mlp,
        thresholds=THRESHOLDS,
        embedding_dict=embedding_dict,
        vector_size=loaded_dim,
        num_layers=NUM_LAYERS
    )
    
    retraining_strategy = partial(
        mlp.retrain_static_generic_mlp,
        embedding_dict=embedding_dict,
        vector_size=loaded_dim,
        num_layers=NUM_LAYERS
    )
    
    # C. Run Manager
    fold_results = metrics.run_nested_cv_experiment(
        X=X_raw_df, 
        df_target=df_target,
        fold_files=fold_files,
        tuning_callback=tuning_strategy,
        retraining_callback=retraining_strategy,
        model_name=f"MLP_{NUM_LAYERS}Layer",
        feature_name=f"Kane_{dim}", 
        cohort_name=COHORT_NAME
    )
    
    for res in fold_results:
        res['Dimension'] = dim
        res['Feature_Type'] = "Kane"
        res['Layers'] = NUM_LAYERS
        
    all_results.extend(fold_results)

# SAVE & VISUALIZE
print(f"\nSaving results to: {RESULTS_DIR}")

history_log = []
for res in all_results:
    if 'History' in res and res['History'] is not None:
        # Convert the raw history dict into a DataFrame
        df_hist = pd.DataFrame(res['History'])
        df_hist['Fold'] = res['Fold']
        df_hist['Dimension'] = res['Dimension']
        df_hist['Epoch'] = range(1, len(df_hist) + 1)
        history_log.append(df_hist)

# Pass history_log to the save function
visualization.save_experiment_results(all_results, history_log, f"{EXPERIMENT_NAME}_metrics", RESULTS_DIR, PLOTS_DIR)
visualization.save_aggregated_validation_curves(history_log, PLOTS_DIR / 'stability_checks')
visualization.summarize_experiment(all_results, RESULTS_DIR, PLOTS_DIR, group_by=['Model', 'Feature_Type', 'Dimension'], script_name=EXPERIMENT_NAME)

print("Experiment Complete.")
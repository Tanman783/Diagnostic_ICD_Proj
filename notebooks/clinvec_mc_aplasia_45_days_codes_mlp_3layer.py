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
from src.utils.training import mlp

# 2. CONFIGURATION
EXPERIMENT_NAME = "clinvec_mc_aplasia_45_days_codes_mlp_3layer"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes"
NUM_LAYERS = 3
# ClinVec File
EMBEDDING_FILE = project_root / 'data' / 'embeddings' / 'ClinVec_icd10cm_embeddings.csv'

# Hyperparameters
THRESHOLDS = [0.0, 0.001, 0.005, 0.01]

# Paths
DATA_DIR = project_root / 'data' / COHORT_NAME
FOLDS_DIR = DATA_DIR / 'cv_folds'
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

print("Converting patient history to sequences...")
sequences_dict = base.prepare_sequences(df_codes)
X_raw_df, df_target = base.format_and_align_data(sequences_dict, df_target, valid_ids)

print(f"Aligned Data. Patients: {len(X_raw_df)}. Folds: {len(fold_files)}")

# 4. LOAD EMBEDDINGS (Lookup Dictionary)
print(f"\nLoading ClinVec dictionary: {EMBEDDING_FILE.name}")
if not EMBEDDING_FILE.exists():
    print(f"[ERROR] Embedding file not found: {EMBEDDING_FILE}")
    sys.exit(1)

embedding_dict, loaded_dim = external.load_pretrained_embeddings(EMBEDDING_FILE)

if not embedding_dict:
    print("[ERROR] Failed to load embedding dictionary.")
    sys.exit(1)

print(f"Loaded Dictionary: {len(embedding_dict)} codes, Dimension: {loaded_dim}")

# 5. TUNING
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

# 6. RUN EXPERIMENT
all_results = []

fold_results = metrics.run_nested_cv_experiment(
    X=X_raw_df, 
    df_target=df_target,
    fold_files=fold_files,
    tuning_callback=tuning_strategy,
    retraining_callback=retraining_strategy,
    model_name=f"MLP_{NUM_LAYERS}Layer",
    feature_name="ClinVec", 
    cohort_name=COHORT_NAME
)

# Inject Meta-data
for res in fold_results:
    res['Dimension'] = loaded_dim
    res['Feature_Type'] = "ClinVec"
    res['Source'] = "Harvard_ClinGraph"
    res['Layers'] = NUM_LAYERS
    
all_results.extend(fold_results)

# 7. SAVE & VISUALIZE
print(f"\nSaving results to: {RESULTS_DIR}")

history_log = []
for res in all_results:
    if 'History' in res and res['History'] is not None:
        df_hist = pd.DataFrame(res['History'])
        df_hist['Fold'] = res['Fold']
        df_hist['Dimension'] = res['Dimension']
        df_hist['Epoch'] = range(1, len(df_hist) + 1)
        history_log.append(df_hist)

visualization.save_experiment_results(all_results, history_log, f"{EXPERIMENT_NAME}_metrics", RESULTS_DIR, PLOTS_DIR)
visualization.summarize_experiment(all_results, RESULTS_DIR, PLOTS_DIR, group_by=['Model', 'Feature_Type', 'Dimension'], script_name=EXPERIMENT_NAME)
visualization.save_aggregated_validation_curves(history_log, PLOTS_DIR / 'learning_curves')


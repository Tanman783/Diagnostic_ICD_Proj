import sys
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial
from gensim.models import Word2Vec

# 1. SETUP PATHS & IMPORTS
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils import data_loader
from src.utils.evaluation import metrics, visualization
from src.utils.preprocessing.embeddings import base 
from src.utils.training import mlp

# 2. CONFIGURATION
EXPERIMENT_NAME = "w2v_emb_mc_aplasia_45_days_codes_mlp2layer"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes"
NUM_LAYERS = 2  # 2-LAYER ARCHITECTURE

# Thresholds & Dimensions
THRESHOLDS = [0.0, 0.001, 0.005, 0.01]
DIMENSIONS = [10, 50, 100, 1000]

# Paths
DATA_DIR = project_root / 'data' / COHORT_NAME
FOLDS_DIR = DATA_DIR / 'cv_folds'

RESULTS_DIR = project_root / 'results' / 'experiments' / CODE_TYPE / EXPERIMENT_NAME
PLOTS_DIR = project_root / 'results' / 'plots' / CODE_TYPE / EXPERIMENT_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. WRAPPER FOR GENSIM WORD2VEC

def train_w2v_wrapper(sentences, vector_size):
    """
    train Word2Vec.
    """
    if not sentences:
        return None
        
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,  
        min_count=1,
        workers=4,
        epochs=10,
        sg=1
    )
    return model.wv

# 4. LOAD & PREPARE DATA
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

# 5. EXPERIMENT LOOP
all_results = []

for dim in DIMENSIONS:
    print(f"\n{'='*40}")
    print(f"Processing W2V Dimension {dim} (MLP {NUM_LAYERS}-Layer)")
    print(f"{'='*40}")

    # A. Define Tuning Strategy (Inject params into mlp.tune_dynamic_generic_mlp)
    tuning_strategy = partial(
        mlp.tune_dynamic_generic_mlp,
        thresholds=THRESHOLDS,
        vector_size=dim,
        embedding_trainer_fn=train_w2v_wrapper,
        num_layers=NUM_LAYERS
    )
    
    # B. Define Retraining Strategy (Inject params into mlp.retrain_dynamic_generic_mlp)
    retraining_strategy = partial(
        mlp.retrain_dynamic_generic_mlp,
        vector_size=dim,
        embedding_trainer_fn=train_w2v_wrapper,
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
        feature_name=f"W2V_{dim}", 
        cohort_name=COHORT_NAME
    )
    
    # Inject Meta-data
    for res in fold_results:
        res['Dimension'] = dim
        res['Feature_Type'] = "Word2Vec"
        res['Layers'] = NUM_LAYERS
        
    all_results.extend(fold_results)

# 6. SAVE & VISUALIZE
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
visualization.save_aggregated_validation_curves(history_log, PLOTS_DIR / 'stability_checks')
visualization.summarize_experiment(all_results, RESULTS_DIR, PLOTS_DIR, group_by=['Model', 'Feature_Type', 'Dimension'], script_name=EXPERIMENT_NAME)

print("Experiment Complete.")
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
from src.utils.preprocessing.embeddings import base, node2vec
from src.utils.training import mlp

# 2. CONFIGURATION

EXPERIMENT_NAME = "node2vec_emb_mc_aplasia_45_days_codes_mlp_3layer"
COHORT_NAME = "mimic_cohort_aplasia_45_days"
CODE_TYPE = "icd_codes"
NUM_LAYERS = 3

# Thresholds & Dimensions
THRESHOLDS = [0.0, 0.001, 0.005, 0.01]
DIMENSIONS = [10, 50, 100, 1000]

# Node2Vec Hyperparameters
NUM_WALKS = 10
WALK_LENGTH = 20
P = 1.0 
Q = 1.0
WINDOW = 10

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

# 4. EXPERIMENT LOOP

all_results = []

for dim in DIMENSIONS:
    print(f"\n{'='*50}")
    print(f"Processing Node2Vec Dimension {dim} (MLP {NUM_LAYERS}-Layer)")
    print(f"{'='*50}")   
    # A. Generate Static Embeddings from Graph
    print(f"Generating Node2Vec embeddings (Dim={dim}, Walks={NUM_WALKS})...")
    try:
        embedding_dict = node2vec.train_node2vec(
            sentences=None, 
            vector_size=dim, 
            num_walks=NUM_WALKS, 
            walk_length=WALK_LENGTH,
            p=P,
            q=Q,
            window=WINDOW,
            workers=4
        )
        print(f" > Embeddings ready. Count: {len(embedding_dict)}")
    except Exception as e:
        print(f"[ERROR] Node2Vec generation failed: {e}")
        continue

    # B. Define Tuning Strategy
    tuning_strategy = partial(
        mlp.tune_static_generic_mlp, 
        thresholds=THRESHOLDS, 
        embedding_dict=embedding_dict, 
        vector_size=dim,
        num_layers=NUM_LAYERS
    ) 
    # C. Define Retraining Strategy
    retraining_strategy = partial(
        mlp.retrain_static_generic_mlp, 
        embedding_dict=embedding_dict, 
        vector_size=dim,
        num_layers=NUM_LAYERS
    )
    # D. Run Manager
    fold_results = metrics.run_nested_cv_experiment(
        X=X_raw_df, 
        df_target=df_target,
        fold_files=fold_files,
        tuning_callback=tuning_strategy,
        retraining_callback=retraining_strategy,
        model_name=f"MLP_{NUM_LAYERS}Layer",
        feature_name=f"Node2Vec_{dim}", 
        cohort_name=COHORT_NAME
    )
    # Inject Meta-data
    for res in fold_results:
        res['Dimension'] = dim
        res['Feature_Type'] = "Node2Vec"
        res['Layers'] = NUM_LAYERS
        res['Walk_Length'] = WALK_LENGTH
        res['Num_Walks'] = NUM_WALKS
        res['P'] = P
        res['Q'] = Q
        res['Window'] = WINDOW        
    all_results.extend(fold_results)
    
# 5. SAVE & VISUALIZE

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
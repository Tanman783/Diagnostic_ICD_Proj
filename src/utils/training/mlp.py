import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from src.utils.preprocessing.embeddings import base

# 1. PYTORCH MODEL DEF (GENERIC MULTI-LAYER)

class GenericMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128], dropout_rate=0.3):
        """
        Args:
            input_dim (int): Size of input vector.
            hidden_layers (list): List of hidden layer sizes (e.g., [128, 64]).
        """
        super(GenericMLP, self).__init__()
        
        layers_list = []
        prev_dim = input_dim
        
        # Hidden Layers
        for h_dim in hidden_layers:
            layers_list.append(nn.Linear(prev_dim, h_dim))
            layers_list.append(nn.BatchNorm1d(h_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Output Layer (Binary)
        layers_list.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.network(x)

# 2. SKLEARN-COMPATIBLE WRAPPER

class ScaledGenericMLP(BaseEstimator, ClassifierMixin):
    """
    Wraps Generic PyTorch Model + StandardScaler + Imbalance Handling (pos_weight).
    """
    def __init__(self, input_dim, hidden_layers=[128], epochs=50, batch_size=32, learning_rate=0.001, device=None, verbose=False):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Initialize Generic Model
        self.model = GenericMLP(input_dim, hidden_layers).to(self.device)
        self.scaler = StandardScaler()
        self.fitted = False
        self.classes_ = [0, 1]
        
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self, X, y, X_val=None, y_val=None, patience=10):
        self.history = {'train_loss': [], 'val_loss': []}
        
        # 1. Scale Data
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Validation Prep
        val_loader = None
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

        # 2. Handle Imbalance
        num_pos = y.sum()
        num_neg = len(y) - num_pos
        pos_weight = torch.tensor(num_neg / (num_pos + 1e-5), dtype=torch.float32).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 3. Train Loop
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = self.epochs 
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
            epoch_loss = running_loss / len(dataset)
            self.history['train_loss'].append(epoch_loss)
            
            # Validation & Early Stopping
            if val_loader:
                self.model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for bx, by in val_loader:
                        out = self.model(bx)
                        loss = criterion(out, by)
                        val_running_loss += loss.item() * bx.size(0)
                
                val_loss = val_running_loss / len(val_loader.dataset)
                self.history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose: print(f"Early stop at {epoch+1}")
                        break
        
        self.fitted = True
        return best_epoch

    def predict_proba(self, X):
        if not self.fitted: raise Exception("Model not fitted!")
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack((1 - probs, probs))

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

# 3. EMBEDDING WRAPPER

class GenericMLPEmbeddingWrapper:
    def __init__(self, scaled_model, embedding_dict, valid_codes, vector_size):
        self.model = scaled_model
        self.embedding_dict = embedding_dict
        self.valid_codes = valid_codes
        self.vector_size = vector_size
        self.classes_ = [0, 1]

    def predict(self, X):
        X_vec = base.filter_and_vectorize(X, self.valid_codes, self.embedding_dict, self.vector_size)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        X_vec = base.filter_and_vectorize(X, self.valid_codes, self.embedding_dict, self.vector_size)
        return self.model.predict_proba(X_vec)

# 4. HELPER: DYNAMIC LAYERING LOGIC

def calculate_dynamic_layers(input_dim, num_layers):
    """
    Implements the Bottleneck Rule:
    Layer 1: min(128, input_dim)
    Subsequent: Previous // 2
    """
    layers = []
    current_dim = min(128, input_dim)
    layers.append(current_dim)
    
    for _ in range(num_layers - 1):
        current_dim = current_dim // 2
        # Ensure minimum size of 4 neurons to prevent collapse
        current_dim = max(4, current_dim) 
        layers.append(current_dim)
        
    return layers

# 5. TUNING STRATEGIES (STATIC)

def tune_static_generic_mlp(X_dev, y_dev, thresholds, embedding_dict, vector_size, num_layers=2):
    """
    Finds best Threshold AND Average Best Epochs using 3-Fold Inner CV.
    """
    best_avg_auc = -1
    best_params = {'Selected_Threshold': thresholds[0], 'Best_Epoch': 30, 'History': None}
    
    # Calculate Dynamic Layers based on Vector Size
    hidden_layers = calculate_dynamic_layers(vector_size, num_layers)
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for t in thresholds:
        fold_scores = []
        fold_epochs = []
        representative_history = None
        
        # Inner Loop
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            X_inner_train, X_inner_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_inner_train, y_inner_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            # 1. Filter Logic (Inner Train)
            all_codes = [c for sublist in X_inner_train.iloc[:,0] for c in sublist]
            code_counts = Counter(all_codes)
            min_count = int(len(X_inner_train) * t)
            valid_codes = {c for c, count in code_counts.items() if count >= min_count}
            
            # 2. Vectorize
            X_vec_train = base.filter_and_vectorize(X_inner_train, valid_codes, embedding_dict, vector_size)
            X_vec_val = base.filter_and_vectorize(X_inner_val, valid_codes, embedding_dict, vector_size)
            
            # 3. Train with Early Stopping
            model = ScaledGenericMLP(input_dim=vector_size, hidden_layers=hidden_layers, epochs=50) 
            optimal_epochs = model.fit(X_vec_train, y_inner_train, X_vec_val, y_inner_val, patience=10)
            
            fold_epochs.append(optimal_epochs)
            representative_history = model.history

            # 4. Score
            probs_val = model.predict_proba(X_vec_val)[:, 1]
            try:
                score = roc_auc_score(y_inner_val, probs_val)
            except:
                score = 0.5
            fold_scores.append(score)
            
        # Averages
        avg_score = np.mean(fold_scores)
        avg_epochs = int(np.mean(fold_epochs))
            
        if avg_score > best_avg_auc:
            best_avg_auc = avg_score
            best_params['Selected_Threshold'] = t
            best_params['Best_Epoch'] = avg_epochs
            best_params['History'] = representative_history
            
    best_params['Val_AUC_Best'] = best_avg_auc
    return best_params

def retrain_static_generic_mlp(X_combined, y_combined, best_params, embedding_dict, vector_size, num_layers=2):
    thresh = best_params.get('Selected_Threshold', 0.0)
    epochs = best_params.get('Best_Epoch', 30)
    
    # Calculate Dynamic Layers
    hidden_layers = calculate_dynamic_layers(vector_size, num_layers)
    
    all_codes = [c for sublist in X_combined.iloc[:,0] for c in sublist]
    code_counts = Counter(all_codes)
    min_count = int(len(X_combined) * thresh)
    valid_codes = {c for c, count in code_counts.items() if count >= min_count}
    
    X_vec_combined = base.filter_and_vectorize(X_combined, valid_codes, embedding_dict, vector_size)
    
    final_model = ScaledGenericMLP(input_dim=vector_size, hidden_layers=hidden_layers, epochs=epochs)
    final_model.fit(X_vec_combined, y_combined, patience=None)
    if best_params.get('History') is not None:
        final_model.history = best_params['History']
    
    wrapper = GenericMLPEmbeddingWrapper(final_model, embedding_dict, valid_codes, vector_size)
    return wrapper, X_combined.columns

# 6. TUNING STRATEGIES (DYNAMIC - W2V/NODE2VEC ETC.)

def tune_dynamic_generic_mlp(X_dev, y_dev, thresholds, vector_size, embedding_trainer_fn, num_layers=2):
    """
    Finds best Threshold AND Average Best Epochs using 3-Fold Inner CV. Also trains Embeddings internally.
    """
    best_avg_auc = -1
    best_params = {'Selected_Threshold': thresholds[0], 'Best_Epoch': 30, 'History': None}
    
    # Calculate Dynamic Layers
    hidden_layers = calculate_dynamic_layers(vector_size, num_layers)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for t in thresholds:
        fold_scores = []
        fold_epochs = [] 
        representative_history = None

        # Inner Loop 
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            X_inner_train, X_inner_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_inner_train, y_inner_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            # A. Filter Logic (Inner Train)
            all_codes = [c for sublist in X_inner_train.iloc[:,0] for c in sublist]
            code_counts = Counter(all_codes)
            min_count = int(len(X_inner_train) * t)
            valid_codes = {c for c, count in code_counts.items() if count >= min_count}
            
            # B. Train Dynamic Embedding (INNER TRAIN ONLY)
            train_sentences = [
                [c for c in row if c in valid_codes] 
                for row in X_inner_train.iloc[:,0]
            ]
            embedding_kv = embedding_trainer_fn(train_sentences, vector_size)
            
            # C. Vectorize
            X_vec_train = base.filter_and_vectorize(X_inner_train, valid_codes, embedding_kv, vector_size)
            X_vec_val = base.filter_and_vectorize(X_inner_val, valid_codes, embedding_kv, vector_size)
            
            # D. Train MLP
            model = ScaledGenericMLP(input_dim=vector_size, hidden_layers=hidden_layers, epochs=50)
            optimal_epochs = model.fit(X_vec_train, y_inner_train, X_vec_val, y_inner_val, patience=10)
            
            fold_epochs.append(optimal_epochs)
            representative_history = model.history

            # E. Evaluate
            probs_val = model.predict_proba(X_vec_val)[:, 1]
            try:
                score = roc_auc_score(y_inner_val, probs_val)
            except:
                score = 0.5
            fold_scores.append(score)
            
        # Averages
        avg_score = np.mean(fold_scores)
        avg_epochs = int(np.mean(fold_epochs))

        if avg_score > best_avg_auc:
            best_avg_auc = avg_score
            best_params['Selected_Threshold'] = t
            best_params['Best_Epoch'] = avg_epochs
            best_params['History'] = representative_history

    best_params['Val_AUC_Best'] = best_avg_auc
    return best_params

def retrain_dynamic_generic_mlp(X_combined, y_combined, best_params, vector_size, embedding_trainer_fn, num_layers=2):
    thresh = best_params.get('Selected_Threshold', 0.0)
    epochs = best_params.get('Best_Epoch', 30)
    
    # Calculate Dynamic Layers
    hidden_layers = calculate_dynamic_layers(vector_size, num_layers)
    
    all_codes = [c for sublist in X_combined.iloc[:,0] for c in sublist]
    code_counts = Counter(all_codes)
    min_count = int(len(X_combined) * thresh)
    valid_codes = {c for c, count in code_counts.items() if count >= min_count}
    
    train_sentences = [
        [c for c in row if c in valid_codes] 
        for row in X_combined.iloc[:,0]
    ]
    
    embedding_kv = embedding_trainer_fn(train_sentences, vector_size)
    
    X_vec_combined = base.filter_and_vectorize(X_combined, valid_codes, embedding_kv, vector_size)
    
    final_model = ScaledGenericMLP(input_dim=vector_size, hidden_layers=hidden_layers, epochs=epochs)
    final_model.fit(X_vec_combined, y_combined, patience=None)
    if best_params.get('History') is not None:
        final_model.history = best_params['History']
    wrapper = GenericMLPEmbeddingWrapper(final_model, embedding_kv, valid_codes, vector_size)
    return wrapper, X_combined.columns
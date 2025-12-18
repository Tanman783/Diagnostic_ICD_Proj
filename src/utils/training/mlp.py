import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import copy
import random
import os

def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleMLP(nn.Module):
    """
    A dynamic Multi-Layer Perceptron in PyTorch.
    """
    def __init__(self, input_dim, layers, dropout_rate):
        super(SimpleMLP, self).__init__()
        
        layer_list = []
        
        # Input Layer to First Hidden Layer
        prev_dim = input_dim
        for hidden_dim in layers:
            layer_list.append(nn.Linear(prev_dim, hidden_dim))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Output Layer (Binary Classification -> 1 neuron)
        layer_list.append(nn.Linear(prev_dim, 1))
        layer_list.append(nn.Sigmoid())
        # Pack everything into a sequential model
        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)
    

def train_mlp_model(X_train, y_train, input_dim, X_val=None, y_val=None, layers=[128, 64], dropout=0.2, learning_rate=0.001, epochs=30, batch_size=32, patience=10):
    """
    Wrapper for PyTorch MLP training with Early Stopping AND Class Balancing.
    """
    # convert pd to np        
    if isinstance(X_train, (pd.DataFrame, pd.Series)): X_train = X_train.values
    if isinstance(y_train, (pd.DataFrame, pd.Series)): y_train = y_train.values
    if X_val is not None and isinstance(X_val, (pd.DataFrame, pd.Series)): X_val = X_val.values
    if y_val is not None and isinstance(y_val, (pd.DataFrame, pd.Series)): y_val = y_val.values
        
    # Setup Device
    # Setup Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # Prepare Data & Handle Imbalance
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Handle Class Imbalance with WeightedRandomSampler
    # Calculate weights for each sample so rare classes are picked more often
    targets = y_train.flatten().astype(int)
    class_counts = np.bincount(targets)
    
    # Handle edge case where a fold might have 0 positives (rare but possible)
    if len(class_counts) < 2: 
        class_weights = np.ones(len(targets))
    else:
        # Inverse frequency: Rare class gets high weight, Common class gets low weight
        weight_per_class = 1. / np.maximum(class_counts, 1) 
        sample_weights = torch.tensor([weight_per_class[t] for t in targets], dtype=torch.float32)

    # Create Sampler
    # replacement=True allows us to re-sample the rare patients multiple times per epoch
    if len(class_counts) >= 2:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        # Note: shuffle must be False when using a sampler
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare Validation Data (if provided)
    val_data = None
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        val_data = (X_val_tensor, y_val_tensor)
    
    # Initialize Model
    model = SimpleMLP(input_dim, layers, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)

        # Validate & Early Stopping
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(val_data[0])
                val_loss = criterion(val_preds, val_data[1]).item()
                history['val_loss'].append(val_loss)
            
            # Check patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped_epoch = epoch + 1
                    model.load_state_dict(best_model_wts) # Restore best model
                    break
        else:
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    history['stopped_epoch'] = stopped_epoch
    # Return model to CPU. This is CRITICAL.
    # If we leave it on GPU, the evaluation code later (sklearn) will crash.
    return model.to('cpu'), history
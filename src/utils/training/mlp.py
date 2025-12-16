import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def train_mlp_model(X_train, y_train, input_dim, layers=[128, 64], dropout=0.2, learning_rate=0.001, epochs=20, batch_size=32):
    """
    Wrapper for PyTorch MLP training.
    """
    print(f"Building PyTorch MLP: Input({input_dim}) -> {layers} -> Output(1)")
    # convert pd to np
    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        X_train = X_train.values
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train = y_train.values
        
    # Setup Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Prepare Data
    # Convert Numpy -> PyTorch Tensors (Float32 is required for NNs)
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    # Target needs to be shape (N, 1) for BCELoss. 
    # If y is just [0, 1, 0], unsqueeze makes it [[0], [1], [0]].
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    model = SimpleMLP(input_dim, layers, dropout).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy (Standard for 0/1 classification)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    history = {'loss': []}
    print("Starting MLP Training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("PyTorch MLP Training Complete.")
    
    # Return model to CPU. This is CRITICAL.
    # If we leave it on GPU, the evaluation code later (sklearn) will crash.
    return model.to('cpu'), history
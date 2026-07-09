import torch
import torch.nn as nn 
import numpy
from data import get_market_data
from features import build_and_label
import numpy as np
from signals import get_signals
import cupy as cp
def learning(truth,features,validation_truth,validation_features):
    
   
    mean = features.mean(axis=0)   # per-column mean (axis=0 → one per feature)
    std  = features.std(axis=0)    # per-column std
    features = normalize(features , mean , std)
    validation_features = normalize(validation_features , mean , std)
    features = torch.FloatTensor(features)
    truth = torch.FloatTensor(truth).unsqueeze(1)
    validation_features = torch.FloatTensor(validation_features)
    validation_truth = torch.FloatTensor(validation_truth).unsqueeze(1)
    
    
    model = nn.Sequential(
        nn.Linear(4, 8),   # layer 1: 4 inputs → 8 neurons
        nn.ReLU(),         # activation
        nn.Linear(8, 1),   # layer 2: 8 inputs → 1 output
        nn.Sigmoid()       # final activation
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss() 
    for epoch in range(500):
        pred = model(features)           # forward pass — replaces your predict()
        loss = loss_fn(pred, truth)     # compute loss
        
        optimizer.zero_grad()              # clear old gradients
        loss.backward()                    # backprop — computes ALL gradients automatically
        optimizer.step()                   # update all weights
        
        if epoch % 100 == 0:
            print(f"epoch {epoch} loss: {loss.item():.8f}")
    with torch.no_grad():
       
    
        pred_new = model(validation_features)           # forward pass — replaces your predict()
        loss_new = loss_fn(pred_new, validation_truth) 
    
    print(f"train loss {loss.item():.4f} (prob {np.exp(-loss.item()):.3f})  |  "
        f"val loss {loss_new.item():.4f} (prob {np.exp(-loss_new.item()):.3f})")
    return model
def normalize(features, mean, std):
    return (features - mean) / std    
def run_pipeline(start_year,end_year,start_year_validation,end_year_validation):
    train_kandles=get_market_data(start_year,end_year)
    validation_kandles=get_market_data(start_year_validation,end_year_validation)
    train_kandles_signals=cp.array(train_kandles.prices_close, dtype=cp.float32)
    validation_kandles_signals=cp.array(validation_kandles.prices_close, dtype=cp.float32)
    signals_train_gpu=get_signals(train_kandles_signals,rangeL=range(50,51))
    signals_validation_gpu=get_signals(validation_kandles_signals,rangeL=range(50,51))
    sig_idx_train = signals_train_gpu.map_of_signals[int(50)][int(20)]
    sig_idx_validation = signals_validation_gpu.map_of_signals[int(50)][int(20)]
    signal_train    =   signals_train_gpu.matrix[sig_idx_train].get()
    signal_validation   = signals_validation_gpu.matrix[sig_idx_validation].get()
    
    features,truth=build_and_label(train_kandles,signal_train)
    
    validation_features,validation_truth=build_and_label(validation_kandles,signal_validation)
    
    model=learning(truth,features,validation_truth,validation_features)
    return model
if __name__ == "__main__":
    model = run_pipeline(2018, 2024, 2025, 2026)
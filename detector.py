import torch
import torch.nn as nn 
import numpy
from data import get_market_data
from features import build_and_label
import numpy as np
def learning(truth,features,validation_truth,validation_features):
    
    print("=== INTO learning ===")
    print("features (numpy in):", features.shape, "truth:", truth.shape)
    mean = features.mean(axis=0)   # per-column mean (axis=0 → one per feature)
    std  = features.std(axis=0)    # per-column std
    features = (features - mean) / std
    validation_features = (validation_features - mean) / std 
    features = torch.FloatTensor(features)
    truth = torch.FloatTensor(truth).unsqueeze(1)
    validation_features = torch.FloatTensor(validation_features)
    validation_truth = torch.FloatTensor(validation_truth).unsqueeze(1)
    print("features tensor:", features.shape, "truth tensor:", truth.shape)
    print("val_features tensor:", validation_features.shape, "val_truth tensor:", validation_truth.shape)
    
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
def run_pipeline(start_year,end_year,start_year_validation,end_year_validation):
    train_kandles=get_market_data(start_year,end_year)
    validation_kandles=get_market_data(start_year_validation,end_year_validation)
    features,truth=build_and_label(train_kandles)
    print(">>> after build_and_label (train): features", features.shape, "truth", truth.shape)
    validation_features,validation_truth=build_and_label(validation_kandles)
    print(">>> after build_and_label (val): features", validation_features.shape, "truth", validation_truth.shape)
    model=learning(truth,features,validation_truth,validation_features)
    return model
if __name__ == "__main__":
    model = run_pipeline(2018, 2024, 2025, 2026)
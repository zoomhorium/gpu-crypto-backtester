import numpy as np

import pandas as pd
from adx import prepare_ai_data
import numpy as np
from types import SimpleNamespace
def compute_volatility(closes, window=20):
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    rolling_std = pd.Series(returns).rolling(window).std().to_numpy()
    return np.concatenate([[np.nan], rolling_std])  

def compute_volume_ratio(volume,window=20):
    volume_ratio=volume/(pd.Series(volume).rolling(window).mean().to_numpy())
    return volume_ratio
def build_features(kandles, window_volume, window_volatility, window_adx):
    volatility = compute_volatility(kandles.prices_close, window_volatility)
    volume     = compute_volume_ratio(kandles.volume, window_volume)
    indicators = prepare_ai_data(window_adx, kandles.prices_close, kandles.highs, kandles.lows)
    adx = indicators.adx[window_adx]
    div = indicators.division[window_adx]
             # <-- add
    features = np.column_stack([volatility, volume, adx, div])
    
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
      # <-- add
    return features, mask
def finding_winner(closes, highs, lows, volatilitys, multiplier=2.0):
    truth = np.full(len(closes), np.nan)
    for idx in range(len(closes)):
        if np.isnan(volatilitys[idx]):
            continue
        barrier = multiplier * volatilitys[idx] * closes[idx]
        upper_limit = closes[idx] + barrier
        lower_limit = closes[idx] - barrier
        c=0;
        for future_high,future_low in  zip(highs[idx+1:], lows[idx+1:]):
            
            if future_high>upper_limit:
                truth[idx]=1
                break
            if future_low<lower_limit:
                truth[idx]=1
                break
            if c>=20:
                truth[idx]=0
                break
            c+=1
    return truth
def finding_winner_signal(closes, signal, highs, lows, volatilitys, multiplier=2.0):
    truth = np.full(len(closes), np.nan)
    position=0
    entry=0
    for idx in range(len(closes)):
        if signal[idx]==0:
            continue
        if np.isnan(volatilitys[idx]):
            continue
        
        barrier = multiplier * volatilitys[idx] * closes[idx]

        if signal[idx]==1:
            take = closes[idx] + barrier
            position=1
            stop = closes[idx] - barrier
        if signal[idx]==-1:
            stop = closes[idx] + barrier
            position=-1
            take = closes[idx] - barrier
        
        
        for future_high,future_low in  zip(highs[idx+1:], lows[idx+1:]):
            
            if (future_high>take and position==1) or (future_low<take and position==-1):
                truth[idx]=1
                break
            if (future_low<stop and position==1) or (future_high>stop and position==-1):
                truth[idx]=0
                break
            
    return truth
def build_and_label(market,signal=None, window=20, adx_n=14, multiplier=8.0):
    X, feat_mask = build_features(market, window, window, adx_n)   # <-- uses your existing function
    vol = compute_volatility(market.prices_close, window)
    if signal is None:
        y = finding_winner(market.prices_close, market.highs, market.lows, vol, multiplier)
    else:
        y = finding_winner_signal(market.prices_close, signal, market.highs, market.lows, vol, multiplier)
    y_feat = y[feat_mask]        # align labels to feature rows
    keep = ~np.isnan(y_feat)     # drop rows with no real label
    return X[keep], y_feat[keep] # matched (X, y)

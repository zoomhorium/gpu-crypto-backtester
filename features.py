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
    print("lengths -> vol:", len(volatility), "volume:", len(volume),
          "adx:", len(adx), "div:", len(div))          # <-- add
    features = np.column_stack([volatility, volume, adx, div])
    print("features shape after stack:", features.shape)  # <-- add
    mask = np.isfinite(features).all(axis=1)
    features = features[mask]
    print("features shape after mask:", features.shape)   # <-- add
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
def build_and_label(market, window=20, adx_n=14, multiplier=8.0):
    X, feat_mask = build_features(market, window, window, adx_n)   # <-- uses your existing function
    vol = compute_volatility(market.prices_close, window)
    y = finding_winner(market.prices_close, market.highs, market.lows, vol, multiplier)
    y_feat = y[feat_mask]        # align labels to feature rows
    keep = ~np.isnan(y_feat)     # drop rows with no real label
    return X[keep], y_feat[keep] # matched (X, y)
if __name__ == "__main__":
    # ---- full pipeline test with self-checking asserts ----
    np.random.seed(0)
    n = 60
    closes2 = 60000 + np.cumsum(np.random.randn(n) * 120)
    highs2  = closes2 + np.abs(np.random.randn(n) * 60)
    lows2   = closes2 - np.abs(np.random.randn(n) * 60)
    volume2 = np.abs(np.random.randn(n) * 500 + 2000)
    market = SimpleNamespace(prices_close=closes2, highs=highs2, lows=lows2, volume=volume2)

    # build features + labels
    X, feat_mask = build_features(market, 5, 5, 5)
    vol = compute_volatility(closes2, window=5)
    y = finding_winner(closes2, highs2, lows2, vol, multiplier=2.0)

    # align: keep rows surviving BOTH the feature mask AND having a real label
    y_feat  = y[feat_mask]
    keep    = ~np.isnan(y_feat)
    X_final = X[keep]
    y_final = y_feat[keep]

    # ---- self-checking assertions ----
    assert X_final.shape[0] == y_final.shape[0],        "X and y lengths must match"
    assert not np.isnan(X_final).any(),                 "X must have no NaN"
    assert not np.isnan(y_final).any(),                 "y must have no NaN"
    assert set(np.unique(y_final)).issubset({0.0, 1.0}), "y must be only 0s and 1s"
    assert X_final.shape[1] == 4,                        "X must have 4 feature columns"

    print("ALL CHECKS PASSED")
    print(f"training set: X = {X_final.shape}, y = {y_final.shape}")
    print(f"class balance: {int(y_final.sum())} ones / {len(y_final)} total")
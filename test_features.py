from features import build_features, compute_volatility, finding_winner, finding_winner_signal,compute_volume_ratio, build_and_label
import numpy as np
from types import SimpleNamespace

def make_fake_market(n=60, seed=0):
    np.random.seed(seed)
    closes = 60000 + np.cumsum(np.random.randn(n) * 120)
    highs  = closes + np.abs(np.random.randn(n) * 60)
    lows   = closes - np.abs(np.random.randn(n) * 60)
    volume = np.abs(np.random.randn(n) * 500 + 2000)
    return SimpleNamespace(prices_close=closes, highs=highs, lows=lows, volume=volume)


def test_pipeline_alignment():
    market = make_fake_market()
    X, feat_mask = build_features(market, 5, 5, 5)
    vol = compute_volatility(market.prices_close, window=5)
    y = finding_winner(market.prices_close, market.highs, market.lows, vol, multiplier=2.0)
    y_feat = y[feat_mask]
    keep = ~np.isnan(y_feat)
    X_final = X[keep]
    y_final = y_feat[keep]

    assert X_final.shape[0] == y_final.shape[0],         "X and y lengths must match"
    assert not np.isnan(X_final).any(),                  "X must have no NaN"
    assert not np.isnan(y_final).any(),                  "y must have no NaN"
    assert set(np.unique(y_final)).issubset({0.0, 1.0}), "y must be only 0s and 1s"
    assert X_final.shape[1] == 4,                        "X must have 4 feature columns"
def test_finding_winners_signal():
    closes = np.full(10, 100.0)
    highs  = closes.copy()
    lows   = closes.copy()
    vol    = np.full(10, 0.01)        # barrier = 2*0.01*100 = 2
    highs[3] = 103                     # long at idx2 → wins (1)
    lows[6]  = 97                      # short at idx5 → wins (1)
    lows[9]  = 97                      # long at idx8 → loses (0)
    signal = np.array([0,0,1,0,0,-1,0,0,1,0])
    y_signal = finding_winner_signal(closes, signal, highs, lows, vol, multiplier=2.0)
    
    assert y_signal[2] == 1,      "long that rose should win"
    assert y_signal[5] == 1,      "short that fell should win"
    assert y_signal[8] == 0,      "long that fell should lose"
    assert np.isnan(y_signal[0]), "no-signal candle must be NaN"
def test_finding_winners_vol():
    closes = np.full(25, 100.0)
    highs  = closes.copy()
    lows   = closes.copy()
    vol    = np.full(25, 0.01)        # barrier = 2*0.01*100 = 2
    highs[3] = 109                   # long at idx2 → wins (1)
                          # long at idx8 → loses (0)
    
    y_vol = finding_winner(closes, highs, lows, vol, multiplier=2.0)
    
    
    
    assert y_vol[2]==y_vol[1]==y_vol[0]==1 ,    "result must be  1 before candel 4"
    assert y_vol[3]==0,                        "result must be  0 for fivth candel"
def test_volatility():
    prices = np.array([100., 101, 99, 102, 98, 103, 97])
    out = compute_volatility(prices, window=3)
    assert len(out)==7,                                   "result lenght not equale to input"
    assert np.isnan(out[:3]).all(),                       "warmup is broken"
    assert  np.isfinite(out[3:]).all(),                   "output is no finite"
def test_volume_ratio():
    volume = np.array([1000., 1000, 1000, 2000, 1000, 1000, 1000])
    volume_ratio=compute_volume_ratio(volume,3)
    assert abs(volume_ratio[3] - 1.5) < 1e-9,   "volume_ratio must be 1.5 on index 3"
def test_build_featurs():
    market = make_fake_market()
    features, mask=build_features(market,20,20,14)
    assert features.shape[1] == 4,                         "must have 4 feature columns"
    assert not np.isnan(features).any(),                   "no NaN after masking"
if __name__ == "__main__":
    test_volatility()
    test_finding_winners_signal()
    test_finding_winners_vol()
    test_build_featurs()
    test_pipeline_alignment()
    print("ALL TESTS PASSED")
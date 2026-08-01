from adx import prepare_ai_data
import numpy as np

def test_prepare_ai_data():
    closes = np.array([100.,101,99,102,98,103,97,105,95,108,102,99,101,104,100])
    highs  = closes + 1
    lows   = closes - 1
    n = 5
    result = prepare_ai_data(n, closes, highs, lows)
    adx = result.adx[n]
    div = result.division[n]

    # both arrays must match input length
    assert len(adx) == len(closes),  "adx length must match input"
    assert len(div) == len(closes),  "div length must match input"
    # warmup region must be NaN (the first candle at least — from the prepend)
    assert np.isnan(adx[0]),         "adx warmup must start with NaN"
    assert np.isnan(div[0]),         "div warmup must start with NaN"
    # after warmup, values must be finite (not NaN, not inf)
    assert np.isfinite(adx[n:]).all(),  "adx must be finite after warmup"
def test_adx_perfect_uptrend():
    n = 14
    prices = np.arange(100, 300, dtype=np.float32)   # 200 candles, +1 each
    highs  = prices + 0.5
    lows   = prices - 0.5

    indicators = prepare_ai_data(n, prices, highs, lows)
    adx = indicators.adx[n]
    tail = adx[-50:]        # past the warm-up
    finite = adx[np.isfinite(adx)]          # drop the NaN warm-up
    assert np.allclose(tail, 100), "perfect uptrend must give ADX ≈ 100"
    finite = adx[np.isfinite(adx)]
    assert finite.min() >= 0 and finite.max() <= 100, "ADX must stay within [0, 100]"
if __name__ == "__main__":
    test_prepare_ai_data()
    test_adx_perfect_uptrend()
    print("ALL TESTS PASSED")
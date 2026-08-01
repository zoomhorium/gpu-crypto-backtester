from data import get_market_data
from adx import prepare_gpu_data
from signals import get_signals

md = get_market_data(2018, 2020)
gpu, ind = prepare_gpu_data(range(2, 30), md.prices_close, md.highs, md.lows)
sig = get_signals(gpu.prices, range(6, 300))
print(sig.matrix.shape, sig.matrix.dtype, sig.matrix.nbytes / 1e9, "GB")
import cupy as cp
from models import Signals
def get_signals(data_gpu,rangeL):
    long_sma = {}
    for w in range(2, max(rangeL)+1):
        long_sma[w] = cp.convolve(
            data_gpu,
            cp.ones(w, dtype=cp.float32)/w,
            "valid"
        )

    signal    = {}
    mapOfSignals = {}
    idxs = 0
    for long_window in rangeL:
        mapOfSignals[long_window] = {}
        for short_window in range(2, long_window // 2 + 1):
            d = long_sma[short_window][long_window-short_window:] - long_sma[long_window]
            mapOfSignals[long_window][short_window] = idxs
            signal[idxs] = cp.pad(cp.where(
                (d[:-1] < 0) & (d[1:] >= 0), cp.int8(1),
                cp.where((d[:-1] > 0) & (d[1:] <= 0), cp.int8(-1), cp.int8(0))
            ), (long_window, 0)).astype(cp.int8)
            idxs += 1
    reverseMap = {}
    for long_w, inner in mapOfSignals.items():
        for short_w, idx in inner.items():
            reverseMap[idx] = (long_w, short_w)
    signal_matrix = cp.vstack([signal[i] for i in range(idxs)])
    del signal, long_sma
    cp.get_default_memory_pool().free_all_blocks()
    return Signals(matrix=signal_matrix,reverse_map=reverseMap,map_of_signals=mapOfSignals)
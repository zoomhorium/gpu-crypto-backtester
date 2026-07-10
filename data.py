from repository import get_historical_data_trainning
from datetime import datetime
import cupy as cp
import numpy as np
from models import MarketData

def get_market_data(start,end):
    pricesClose = []
    highs       = []
    lows        = []
    open_time        = []
    volume = []
    for y in range(start, end):
        for i in range(1, 13):
            for start_day, end_day in [(1,8),(8,16),(16,24),(24,None)]:
                start = int(datetime(y, i, start_day).timestamp() * 1000)
                if end_day:
                    end = int(datetime(y, i, end_day).timestamp() * 1000)
                else:
                    end = int(datetime(y, i+1, 1).timestamp() * 1000) if i < 12 else int(datetime(y+1, 1, 1).timestamp() * 1000)
                points = get_historical_data_trainning(symbol="BTCUSDT", startTime=start, endTime=end)
                pricesClose.extend([p["close"] for p in points])
                highs.extend([p["high"]  for p in points])
                lows.extend([p["low"]   for p in points])
                open_time.extend([p["openTime"] for p in points])
                volume.extend([p["volume"] for p in points])
    return MarketData(prices_close=np.array(pricesClose, dtype=np.float32),highs=np.array(highs, dtype=np.float32),lows=np.array(lows, dtype=np.float32),volume=np.array(volume, dtype=np.float32),open_times=np.array(open_time, dtype=np.int64 ),T=len(pricesClose))
def count_combos(ranges):
    total = 0
    
    for long_window in ranges.long_window:
        for short_window in range(2, long_window// 2 + 1):
            total += 1
    total=total*len(ranges.leverage)*len(ranges.stop)*len(ranges.take)*len(ranges.n)*len(ranges.bar)
    return total
def combo_generator(mapOfSignals,_,ranges):
    for leverage in ranges.leverage :
        for stop_div in ranges.stop:
            for take_div in ranges.take:
                for n in ranges.n:
                    for long_window in ranges.long_window:
                        for short_window in range(2, long_window // 2 + 1):
                            sig_idx = mapOfSignals[long_window][short_window]
                            for bar in ranges.bar:
                                yield (n, sig_idx, bar, stop_div, take_div, leverage)
def filtered_combo_generator(mapOfSignals,best_params,_):
    for params in best_params:
        _, n, long_w, short_w, bar, stop, take, lev ,_= params
        sig_idx = mapOfSignals[int(long_w)][int(short_w)]
        yield (int(n), sig_idx, int(bar), int(stop), int(take), int(lev))
def take_chunk(gen, size):
    rows = []
    try:
        for i in range(size):
            rows.append(next(gen))
    except StopIteration:
        pass
    if not rows:
        return None
    data = np.array(rows, dtype=np.int32)
    n_indices_gpu  = cp.asarray(data[:, 0] - 2, dtype=cp.int32)
    bars_gpu       = cp.asarray(data[:, 2],      dtype=cp.int8)
    stop_divs_gpu  = cp.asarray(data[:, 3],      dtype=cp.int32)
    take_divs_gpu  = cp.asarray(data[:, 4],      dtype=cp.int32)
    leverages_gpu = cp.asarray(data[:, 5], dtype=cp.float32)
    signal_idx_gpu = cp.asarray(data[:, 1],      dtype=cp.int32)
    return n_indices_gpu, stop_divs_gpu, take_divs_gpu, leverages_gpu, signal_idx_gpu, bars_gpu

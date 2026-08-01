from data import get_market_data
from adx import prepare_gpu_data
from signals import get_signals
from simulation import run_simulation
import numpy as np
import cupy as cp
import os
np.random.seed(0)
closes = 60000 + np.cumsum(np.random.randn(1000) * 1200)
highs  = closes + np.abs(np.random.randn(1000) * 600)
lows   = closes - np.abs(np.random.randn(1000) * 600)
closes_long = 100000 + np.cumsum(np.random.randn(1000) * 1200)
highs_long   = closes_long  + np.abs(np.random.randn(1000) * 600)
lows_long   = closes_long  - np.abs(np.random.randn(1000) * 600)
closes_short = 40000 + np.cumsum(np.random.randn(1000) * 1200)
highs_short  = closes_short + np.abs(np.random.randn(1000) * 600)
lows_short   = closes_short - np.abs(np.random.randn(1000) * 600)
closes= np.append(closes,closes_long)
closes= np.append(closes,closes_short)
highs = np.append(highs,highs_long)
highs= np.append(highs,highs_short)
lows= np.append(lows,lows_long)
lows= np.append(lows,lows_short)
T=len(closes)
adx=np.full(T,20)
best_n=20 
best_long=20 
best_short=10
best_bar=2 
best_stop=10 
best_take=10 
best_leverage=1
balance_simulation=run_simulation(closes,highs,lows,adx,T,best_n,best_long, best_short,
                   best_bar, best_stop, best_take, best_leverage)
balance_kernel= cp.empty(1, dtype=cp.float32)
out_score= cp.empty(1, dtype=cp.float32)
dir = os.path.dirname(os.path.abspath(__file__))
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
adx=cp.full((1, T), 20.0, dtype=cp.float32)
best_n=cp.array(0, dtype=cp.int32)
best_long=cp.array(20, dtype=cp.int32) 
best_short=cp.array(10, dtype=cp.int32)
best_bar=cp.array(2 , dtype=cp.int8)
best_stop=cp.array(10, dtype=cp.int32)
best_take=cp.array(10 , dtype=cp.int32)
best_leverage=cp.array(1 , dtype=cp.float32)
closes=cp.asarray(closes, dtype=cp.float32)
highs=cp.asarray(highs, dtype=cp.float32)   
lows=cp.asarray(lows, dtype=cp.float32)
module = cp.RawModule(
             code=open(os.path.join(dir, "backtest_kernel.cu")).read(),
             options=("--use_fast_math",),
             name_expressions=["backtest_kernel"]
         )
kernel = module.get_function("backtest_kernel")
blocks  = ((1 + 1023) // 1024,)
signals=get_signals(closes,range(5,21))
kernel(blocks, (1024,), (
                    closes,
                    highs,
                    lows,
                    signals.matrix,
                    adx,
                    best_n, best_stop, best_take,
                    cp.int32(1), cp.int32(T),
                    best_bar, cp.array([signals.map_of_signals[20][10]], dtype=cp.int32), best_leverage, balance_kernel,out_score
                ))
cp.cuda.stream.get_current_stream().synchronize()
print("CPU:", balance_simulation, "GPU:", float(balance_kernel[0]))
print(balance_simulation,T)
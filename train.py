
import cupy as cp
import numpy as np
import time
import os
from paths import DATA_DIR, BASE_DIR
from concurrent.futures import ThreadPoolExecutor
from data import get_market_data
from data import count_combos
from data import combo_generator
from data import filtered_combo_generator
from data import take_chunk
from signals import get_signals
from adx import prepare_gpu_data
from models import BestStrategyData
from models import SurvivorsData
from models import TrainingOutput,NoSurvivorsError
from saving_survivors import save
def prepare_data(start,end,t0,ranges):
    data_cpu =get_market_data(start,end)
    print(f"Training data loaded: {time.time()-t0:.1f}s  ({len(data_cpu.prices_close)} candles)")
    open_times = data_cpu.open_times
    gaps = np.where(np.diff(open_times) > 3600000)[0]
    print(f"Gaps: {gaps}")
    data_gpu,adx=prepare_gpu_data(ranges.n,data_cpu.prices_close,data_cpu.highs,  data_cpu.lows)
    signals=get_signals(data_gpu.prices,ranges.long_window)
    

    return data_cpu,data_gpu,signals,adx
def kernel_loop(start,end,t0,kernel,chunk_size,threshold,variables_fider,params,ranges,progress_callback=None):
        

        startof = time.time()
        
        cpu_data, gpu_data, signals, adx=prepare_data(start,end,t0,ranges)
        
        # # ── Kernel ────────────────────────────────────────────────────────
        num_combos = len(params) if params is not None else count_combos(ranges)
        print(f"Total combinations: {num_combos}")
        if progress_callback:
                    progress_callback(0)
        

        
        print("Kernel compiled OK")
        best_params  = [] 
        best_params_list = []
        best_balance = 0.0
        best_score = 0.0
        
        chunk_start  = 0
        chunk_num    = 0
        gen=variables_fider(signals.map_of_signals,params,ranges)

        with ThreadPoolExecutor(max_workers=1) as executor:
            current_chunk_data = take_chunk(gen, chunk_size)

            while True:
                chunk_num += 1
                t_chunk = time.time()

                n_indices_gpu, stop_divs_gpu, take_divs_gpu, leverages_gpu, signal_idx_gpu, bars_gpu = current_chunk_data
                future = executor.submit(take_chunk, gen, chunk_size)

                n_chunk = len(n_indices_gpu)
                out_balance     = cp.empty(n_chunk, dtype=cp.float32)
                out_score  = cp.empty(n_chunk, dtype=cp.float32)
                blocks  = ((n_chunk + 1023) // 1024,)

                kernel(blocks, (1024,), (
                    gpu_data.prices,
                    gpu_data.highs,
                    gpu_data.lows,
                    signals.matrix,
                    gpu_data.adx_matrix,
                    n_indices_gpu, stop_divs_gpu, take_divs_gpu,
                    cp.int32(n_chunk), cp.int32(gpu_data.T),
                    bars_gpu, signal_idx_gpu, leverages_gpu, out_balance,out_score
                ))
                
                best_val = float(cp.max(out_balance))
                best_score_value = float(cp.max(out_score))
                if np.isfinite(best_score_value) and best_score_value > best_score:
                    best_score = best_score_value
                if np.isfinite(best_val) and best_val > best_balance:
                    best_balance = best_val
                cp.cuda.stream.get_current_stream().synchronize()
                bidx=cp.where(out_balance > threshold)[0]
                values   = cp.asnumpy(out_balance[bidx])
                score   = cp.asnumpy(out_score[bidx])
                n_vals   = cp.asnumpy(n_indices_gpu[bidx])
                sig_vals = cp.asnumpy(signal_idx_gpu[bidx])
                bar_vals = cp.asnumpy(bars_gpu[bidx])
                stop_vals = cp.asnumpy(stop_divs_gpu[bidx])
                take_vals = cp.asnumpy(take_divs_gpu[bidx])
                lev_vals  = cp.asnumpy(leverages_gpu[bidx])
                        
                if len(bidx) > 0:
                    long_short = np.array([signals.reverse_map[int(s)] for s in sig_vals], dtype=np.float32)
                    chunk_data = np.column_stack([
                        values,
                        
                        (n_vals + 2).astype(np.float32),
                        long_short[:, 0],
                        long_short[:, 1],
                        bar_vals.astype(np.float32),
                        stop_vals.astype(np.float32),
                        take_vals.astype(np.float32),
                        lev_vals.astype(np.float32),
                        score
                    ])
                    best_params_list.append(chunk_data)

                del n_indices_gpu, stop_divs_gpu, take_divs_gpu, leverages_gpu, signal_idx_gpu, bars_gpu, out_balance,out_score
                if len(bidx) > 0:
                    del bidx
                print(f"Chunk {chunk_start:>9} / {num_combos} — best balance : {best_balance:.2f} — best score: {best_score:.4f} — params: {len(best_params_list)} — {time.time()-t_chunk:.2f}s")
                chunk_start += n_chunk
                if progress_callback:
                    progress_callback(int(chunk_start/ num_combos*10000))
                current_chunk_data = future.result()
                if current_chunk_data is None:
                    break
        
        best_params = stack_survivors(best_params_list,threshold,start,end)
        
        best_param= max(best_params, key=lambda x: x[8])
        print(f"Survivors: {len(best_params)} — Memory: {best_params.nbytes/1e9:.1f}GB")
        train_data=TrainingOutput(
            market_data=cpu_data,
            best_strategy=BestStrategyData(
                    best_n=best_param[1],
                    best_long=best_param[2],
                    best_short=best_param[3],
                    best_bar=best_param[4],
                    best_stop=best_param[5],
                    best_take=best_param[6],
                    best_leverage=best_param[7],
                    best_balance=best_param[0],
                    best_score=best_param[8]
            ),
            survivors=SurvivorsData(
                survivors=best_params
            ),
            indicators=adx
            ,
            signals=signals) 
        print("\n=== BEST RESULT (TRAINING 2021-2023) ===")
        print(f"Balance: {train_data.best_strategy.best_balance:.2f}")
        print(f"Long Window: {train_data.best_strategy.best_long}, Short Window: {train_data.best_strategy.best_short}")
        print(f"N: {train_data.best_strategy.best_n}, Bar: {train_data.best_strategy.best_bar}")
        print(f"Stop: {train_data.best_strategy.best_stop/10:.1f}%, Take: {train_data.best_strategy.best_take/10:.1f}%, Leverage: {train_data.best_strategy.best_leverage}x")
        print(f"Total time: {time.time()-startof:.1f}s")
        print (sum(len(c) for c in best_params_list))
        return train_data
def train(ranges,progress_callback=None):
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
        if os.path.exists(cuda_path):
            os.add_dll_directory(cuda_path)
        module = cp.RawModule(
             code=open(BASE_DIR / "backtest_kernel.cu").read(),
             options=("--use_fast_math",),
             name_expressions=["backtest_kernel"]
         )
        kernel = module.get_function("backtest_kernel")
        # ── Load TRAINING data 2021-2023 ─────────────────────────────────
        t0 = time.time()
        train_data=kernel_loop(2018,2020,t0,kernel,1750000,170,combo_generator,None,ranges,progress_callback)
        save(DATA_DIR / "round1_params.npy", train_data.survivors.survivors,ranges)


        
        
        round1_params = np.load(DATA_DIR / "round1_params.npy")
            
        
        train_data=kernel_loop(2020,2022,t0,kernel,1750000,170,filtered_combo_generator, round1_params,ranges,progress_callback)
        
        save(DATA_DIR / "round2_params.npy", train_data.survivors.survivors,ranges)
        
        
        round2_params = np.load(DATA_DIR / "round2_params.npy")
            
       
        train_data=kernel_loop(2022,2024,t0,kernel,1750000,170,filtered_combo_generator, round2_params,ranges,progress_callback)
        
        save(DATA_DIR / "round3_params.npy", train_data.survivors.survivors,ranges)
        return train_data
def stack_survivors(best_params_list, threshold, start, end):
    if not best_params_list:
            raise NoSurvivorsError(f"No survivors above balance {threshold} for {start}-{end}")
    best_params = np.vstack(best_params_list)
    return best_params
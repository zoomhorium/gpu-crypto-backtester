import numpy as np
from data import get_market_data
from simulation import run_simulation
from adx import prepare_gpu_data
from train import train 
from models import Ranges,NoSurvivorsError



def pipeline(ranges,progress_callback=None):

    
    train_data = train(ranges,progress_callback)
    
    
        # ── Run validation simulation for top 100 params ──────────────────
    print("\n=== VALIDATION RESULTS (2018-2024) ===")
    pricesClose_train = np.array(
        train_data.market_data.prices_close,
        dtype=np.float32
    )
    highs_train= np.array(
        train_data.market_data.highs,  
        dtype=np.float32
    )
    lows_train= np.array(
        train_data.market_data.lows,
        dtype=np.float32
        
    )
    best_params= train_data.best_strategy
    
    best_n        = int(best_params.best_n)
    best_long     = int(best_params.best_long)
    best_short    = int(best_params.best_short)
    best_bar      = int(best_params.best_bar)
    best_stop     = int(best_params.best_stop)
    best_take     = int(best_params.best_take)
    best_leverage = int(best_params.best_leverage)
    run_simulation(
                pricesClose_train,highs_train,lows_train, train_data.indicators.adx[best_n], train_data.market_data.T,
                best_n, best_long, best_short,
                best_bar, best_stop, best_take, best_leverage
            )
    print("\n=== VALIDATION RESULTS (2024-2025) ===")
    market_data=get_market_data(2024,2026)
    pricesClose_val = np.array(market_data.prices_close)
    T_val = len(pricesClose_val)
    highs_val = np.array(market_data.highs, dtype=np.float32)
    lows_val = np.array(market_data.lows, dtype=np.float32)
    print(f"Validation candles: {T_val}")

    best_params_sorted = sorted(
        train_data.survivors.survivors,
        key=lambda x:x[0],
        reverse=True
    )
    gpu_data,indicator=prepare_gpu_data(ranges.n,pricesClose_val,highs_val,  lows_val)
    for rank, best_param in enumerate(best_params_sorted):
            
            best_n        = int(best_param[1])
            best_long     = int(best_param[2])
            best_short    = int(best_param[3])
            best_bar      = int(best_param[4])
            best_stop     = int(best_param[5])
            best_take     = int(best_param[6])
            best_leverage = int(best_param[7])
            best_balances = int(best_param[0])
            if best_balances == 0:
                continue
            print(f"\nRank {rank+1} — Train balance: {best_balances:.2f} | Params: long={best_long} short={best_short} n={best_n} bar={best_bar} stop={best_stop/10:.1f}% take={best_take/10:.1f}% lev={best_leverage}x")
            run_simulation(
                            pricesClose_val,highs_val,lows_val, indicator.adx[best_n], T_val,
                            best_n, best_long, best_short,
                            best_bar, best_stop, best_take, best_leverage
                        )
    return best_params
if __name__ == "__main__":
    ranges = Ranges(
            leverage=range(1,10),
            stop=range(5,105,5),
            take=range(5,105,5),
            n=range(2,30),
            long_window=range(6,300),
            bar=range(2,30),
        ) 

    
    pipeline(ranges)

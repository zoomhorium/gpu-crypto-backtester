import numpy as np
def run_simulation(prices_cpu,highs_cpu,lows_cpu, adx_cpu, T_sim, best_n, best_long, best_short,
                   best_bar, best_stop, best_take, best_leverage):
    balance   = 100.0
    marge     = 0.0
    entry     = 0.0
    trail     = 0.0
    stop      = 0.0
    take      = 0.0
    position  = 0
    trades    = 0
    stop_f    = best_stop / 1000.0
    take_f    = best_take / 1000.0
    lev       = float(best_leverage)
    signal    = 0
    profit    = 0
    loss      = 0
    diff      = True
    long_sma  = 0.0
    short_sma = 0.0
    trade_count = 0
    maker_fee=0.0002
    taker_fee=0.0004
    slippage = 0.00015

    for i in range(T_sim):
        price = float(prices_cpu[i])
        adx_v = float(adx_cpu[i])
        high=float(highs_cpu[i])
        low=float(lows_cpu[i])
        if i < best_long:
            long_sma += price / best_long
            if i + best_short >= best_long:
                short_sma += price / best_short
                diff = (short_sma > long_sma)
            continue

        if balance <= 1 or not np.isfinite(balance):
            print(f"  Balance wiped at candle {i}")
            print(f"  Winrate: {profit/(profit+loss):.3f}" if profit+loss > 0 else "  No trades completed")
            print(f"  Balance: {balance:.4f}  Trades: {trades}")
            return
        
        

        long_sma  += price / best_long  - prices_cpu[i - best_long]  / best_long
        short_sma += price / best_short - prices_cpu[i - best_short] / best_short

        if  not diff and short_sma >= long_sma and adx_v >= best_bar:
            signal = 1
        elif  diff and short_sma < long_sma and adx_v >= best_bar:
            signal = -1
        else:
            signal = 0
        price_slipp= price+(price*signal*slippage*2)+(price*position*slippage)
        
       
        if (position == 1   ):
            if(entry - low)*marge>=balance:
                balance=0
                print(f"  Balance wiped at candle {i}")
                print(f"  Winrate: {profit/(profit+loss):.3f}" if profit+loss > 0 else "  No trades completed")
                print(f"  Balance: {balance:.4f}  Trades: {trades}")
                return
            elif(trail - stop >= low):
                exit_price = (trail - stop) * (1 - slippage)
                balance  += (exit_price - entry) * marge -exit_price  * marge*taker_fee
            
                
                    
                
                if (exit_price - entry) * marge * position > 0:
                    profit += 1
                else:
                    loss += 1
                stop = take = entry = marge = 0.0
                position = 0
                trade_count = 0
            elif(entry + take <= high):
                balance  += (take ) * marge -(take + entry) * marge*maker_fee
                profit += 1
                stop = take = entry = marge = 0.0
                position = 0
                trade_count = 0
            elif(high > trail):
                    trail = high;
                
        elif (position == -1   ):
            if( high- entry)*marge>=balance:
                balance=0
                print(f"  Balance wiped at candle {i}")
                print(f"  Winrate: {profit/(profit+loss):.3f}" if profit+loss > 0 else "  No trades completed")
                print(f"  Balance: {balance:.4f}  Trades: {trades}")
                return
            elif(trail + stop <= high):
                exit_price = (trail +stop) * (1 + slippage)
                balance  += (exit_price - entry) * marge*-1 -exit_price  * marge*taker_fee
            
                
                    
                
                if (exit_price - entry) * marge * position > 0:
                    profit += 1
                else:
                    loss += 1
                stop = take = entry = marge = 0.0
                position = 0
                trade_count = 0
            elif(entry - take >= low):
                balance  += take  * marge -(entry-take)  * marge*maker_fee
                profit += 1
                stop = take = entry = marge = 0.0
                position = 0
                trade_count = 0 
            elif (low < trail):
                    trail=low;
                
        if(position != signal and signal != 0 and position != 0):
            balance  += (price_slipp - entry) * marge * position-(price_slipp) * marge*taker_fee
            
            if (price - entry) * marge * position > 0:
                profit += 1
            else:
                loss += 1
            stop = take = entry = marge = 0.0
            position = 0
            trade_count = 0
        
        if signal != 0 and trade_count < 8:
            new_marge=balance * lev / price_slipp
            entry    = (entry * marge + price_slipp * new_marge) / (new_marge + marge)
            trail       = entry
            trade_count += 1
            marge += new_marge
            position    = signal
            balance -= entry * new_marge *taker_fee
            stop        = entry * stop_f * (adx_v / 100.0 + 1.0)
            take        = entry * take_f * (adx_v / 100.0 + 1.0)
            trades     += 1

        diff = (short_sma > long_sma)
    if position != 0:
        final_price = prices_cpu[T_sim-1] + (
        prices_cpu[T_sim-1] * (-position) * slippage
        )

        balance += (
        (final_price - entry) * marge * position
        - final_price * marge * taker_fee
        )
    if profit + loss > 0:
        print(f"  Winrate: {profit/(profit+loss):.3f}")
    print(f"  Balance: {balance:.4f}  Trades: {trades} ")
    return balance
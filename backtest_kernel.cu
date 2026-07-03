    typedef signed char int8_t;
    typedef short int16_t;


    __global__ __launch_bounds__(1024) void backtest_kernel(
        float*  __restrict__ prices,
        float*  __restrict__ highs,
        float*  __restrict__ lows,
        int8_t* __restrict__ signal_matrix,
        float* __restrict__ adx_matrix,   // [20, T] shared
        int*    __restrict__ n_indices,     // [num_combos]
        int*    __restrict__ stop_divs,     // [num_combos] — e.g. 5 = 0.5%, 10 = 1.0%
        int*    __restrict__ take_divs,     // [num_combos]
        int     num_combos,
        int     T,
        int8_t*  __restrict__ bars,
        int* __restrict__ signal_gpu,
        float*  __restrict__ leverages,
        
        float*  __restrict__ out_balance
        
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_combos) return;

        float  balance  = 100.0f;
        float  stop     = 0.0f;
        float  take     = 0.0f;
        float  entry    = 0.0f;
        float  marge    = 0.0f;
        int8_t position = 0;
        float  trail    = 0.0f;
        float maker_fee=0.0002;
        float taker_fee=0.0004;
        float slippage = 0.00015;
        float price_slipp = 0;
        int8_t bar  =  bars[idx];
        float leverage = leverages[idx];
        float new_marge = 0;
        
        float exit_price = 0.0f;
        // per-combo stop/take as float percentages: 5 -> 0.005 = 0.5%
        float stop_div = (float)stop_divs[idx] / 1000.0f;
        float take_div = (float)take_divs[idx] / 1000.0f;
        
        
        int8_t trade_count = 0;
        float* my_adx     = adx_matrix + (long long)n_indices[idx] * T;
        int8_t* my_signal  =  signal_matrix+(long long)signal_gpu[idx]*T;
        for (int i = 0; i < T; i++) {
            if (balance <= 3.0f || !isfinite(balance)){
                out_balance[idx] = 0.0f;
                return;}
            float  adx    = (float)my_adx[i];
            int8_t signal  = (adx >= bar) ? my_signal[i] : 0;

            if (signal == 0 && position == 0) continue;
            
            float  price  = prices[i];
            float high =    highs[i];
            float low   = lows[i];
            
            price_slipp= price+(price*signal*slippage*2)+(price*position*slippage);
            
            
            // trailing stop update
           
            
            
            // exit conditions
            if (position == 1){
                
                if ((entry-low)*marge>=balance){
                    out_balance[idx] = 0.0f;
                    return ; 
                }
                else if(trail - stop >= low){
                    trade_count=0;
                    exit_price = (trail - stop) * (1 - slippage);
                    balance  +=  (exit_price-entry) * marge -exit_price * marge*taker_fee;
                    
                    stop      = 0.0f;
                    take      = 0.0f;
                    entry     = 0.0f;
                    marge     = 0.0f;
                    position  = 0;
                }
                else if(entry + take <= high){
                    trade_count=0;
                    
                    balance  +=  take * marge -(entry+take) * marge*maker_fee;
                    
                    stop      = 0.0f;
                    take      = 0.0f;
                    entry     = 0.0f;
                    marge     = 0.0f;
                    position  = 0;
                }
                else if(high > trail){
                    trail = high;
                }
            }
            if (position == -1){
                
                if ((high-entry)*marge>=balance){
                    out_balance[idx] = 0.0f;
                    return ; 
                }
                else if(trail + stop <= high){
                    trade_count=0;
                    exit_price = (trail + stop) * (1 + slippage);
                    balance  +=  (entry-exit_price) * marge -exit_price * marge*taker_fee;
                    
                    stop      = 0.0f;
                    take      = 0.0f;
                    entry     = 0.0f;
                    marge     = 0.0f;
                    position  = 0;
                }
                else if(entry - take >= low){
                    trade_count=0;
                    
                    balance  +=  take * marge -(entry-take) * marge*maker_fee;
                    
                    stop      = 0.0f;
                    take      = 0.0f;
                    entry     = 0.0f;
                    marge     = 0.0f;
                    position  = 0;
                }
                else if (low < trail){
                    trail=low;
                }
            }
            if(position != signal && signal != 0 && position != 0) {
                trade_count=0;
                balance  += (price_slipp - entry) * marge * position-(price_slipp) * marge*taker_fee;
                
                stop      = 0.0f;
                take      = 0.0f;
                entry     = 0.0f;
                marge     = 0.0f;
                position  = 0;
            }
            
            
            // entry
            if (signal!=0 && trade_count < 8) {
                new_marge=balance * leverage / price_slipp;
                entry    = (entry * marge + price_slipp * new_marge) / (new_marge + marge);
                trail    = entry;
                trade_count++;
                
                marge    = marge + new_marge;
                position = signal;
                balance -= entry * new_marge*taker_fee;
                stop     = entry * stop_div * (adx / 100.0f + 1.0f);
                take     = entry * take_div * (adx / 100.0f + 1.0f);
            }
            
            
        }
        if(position != 0){
            float final_price =
                prices[T-1] +
                (prices[T-1] * (-position) * slippage);

            balance +=
                (final_price - entry) * marge * position
                - final_price * marge * taker_fee;
        }
        
        out_balance[idx] = isfinite(balance) ? balance : 0.0f;;
    }

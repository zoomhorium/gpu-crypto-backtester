import numpy as np
from simulation import run_simulation
def test_run_simulation():
   prices=[100,102,104,106,107,100,100]
   highs=[100,102,104,106,107,100,100]
   lows=[100,102,104,106,107,20,20]
   highs_loss=[100,102,104,106,107,105,105]
   lows_loss=[100,102,104,106,107,100,100]
   highs_wiped=[100,102,104,106,107,200,200]
   lows_wiped=[100,102,104,106,107,20,20]
   adx=[20,20,20,20,20,20,20]
   T=len(prices)
   win_trade_balance= run_simulation(prices_cpu=prices,highs_cpu=highs,lows_cpu=lows,adx_cpu=adx,T_sim=T,best_n=14,best_long=5,best_short=2,best_bar=10,best_stop=200,best_take=200,best_leverage=10)
   loss_trade_balance= run_simulation(prices_cpu=prices,highs_cpu=highs_loss,lows_cpu=lows_loss,adx_cpu=adx,T_sim=T,best_n=14,best_long=5,best_short=2,best_bar=10,best_stop=200,best_take=200,best_leverage=10)
   wiped_trade_balance= run_simulation(prices_cpu=prices,highs_cpu=highs_wiped,lows_cpu=lows_wiped,adx_cpu=adx,T_sim=T,best_n=14,best_long=5,best_short=2,best_bar=10,best_stop=200,best_take=200,best_leverage=10)

   
   assert abs(win_trade_balance - 339.448) < 0.01, "winning trade balance changed"
   assert abs(loss_trade_balance-98.74)<0.01,                  "loss trade balance changed"
   assert wiped_trade_balance is None, "wipped trade balance changed"
if __name__ == "__main__":
   test_run_simulation()
   print("ALL SIMULATION TESTS PASSED")
from signals import get_signals
import cupy as cp
def test_signals():
    prices=cp.array([100,102,104,106,107,100,120])
    all_signals=get_signals(prices,range(5,6))
    signals=all_signals.matrix[all_signals.map_of_signals[5][2]]
    assert signals[6]==1,   "there no long signal"
    assert signals[5]==-1,   "there no short signal"
    assert (signals[:5] == 0).all(), "warmup region should be all zeros"
if __name__ == "__main__": 
    
    test_signals()
    print("ALL TESTS PASSED")
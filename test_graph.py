import numpy as np
from models import Grid


from graph import build_grid
def build_grid_slow(data, y_col, x_col, value_col, y_values=None, x_values=None, lower_limit=100, upper_limit=500,balance_col=0):

    if y_values is None:
        step_y=np.gcd.reduce(np.nan_to_num(np.diff(np.unique(data[:, y_col])), nan=0).astype(np.int32))
        if step_y==0:
            step_y=1
        y_values=range(int(np.min(np.unique(data[:,y_col]))),int(np.max(np.unique(data[:,y_col])))+1,step_y) 
    if x_values is None:
        step_x=np.gcd.reduce(np.nan_to_num(np.diff(np.unique(data[:, x_col])), nan=0).astype(np.int32))
        if step_x==0:
            step_x=1
        x_values=range(int(np.min(np.unique(data[:,x_col]))),int(np.max(np.unique(data[:,x_col])))+1,step_x)
    medians = np.full(( len(y_values),len(x_values)), np.nan)
    counts  = np.zeros(( len(y_values),len(x_values)), dtype=int)
    capped_sum  = np.zeros(( len(y_values),len(x_values)), dtype=float)
    p10 = np.full(( len(y_values),len(x_values)), np.nan) 
    p90 = np.full(( len(y_values),len(x_values)), np.nan)
    spread = np.zeros(( len(y_values),len(x_values)), dtype=float)
    drawdown = np.full(( len(y_values),len(x_values)), np.nan)
    
    for i, y in enumerate(y_values):

        for j, x in enumerate(x_values):
            mask = (data[:, y_col] == y) & (data[:, x_col] == x)
            counts[i, j] = mask.sum()
            if counts[i, j]:
                medians[i, j] = np.median(data[mask, value_col])
                drawdown[i,j] = np.median(1 - data[mask, value_col] / data[mask, balance_col])
                p10[i, j],p90[i, j] = np.percentile(data[mask, value_col],[10,90])
                spread[i,j] =( p90[i, j]- p10[i, j])/medians[i, j]
                norm_data = (data[mask, value_col] - lower_limit) / (upper_limit - lower_limit)


                capped_sum[i,j] = np.sum(np.clip(norm_data, 0, 1))
        

    return Grid(y_values=y_values, x_values=x_values, medians=medians, counts=counts , capped_sum=capped_sum ,p10=p10 ,p90=p90 , spread=spread,drawdown = drawdown)
def same(g1, g2):
    return (g1.y_values == g2.y_values
            and g1.x_values == g2.x_values
            and np.array_equal(g1.counts, g2.counts)
            and np.allclose(g1.medians, g2.medians, equal_nan=True)
            and np.allclose(g1.capped_sum, g2.capped_sum, equal_nan=True)
            and np.allclose(g1.p10, g2.p10, equal_nan=True)
            and np.allclose(g1.p90, g2.p90, equal_nan=True)
            and np.allclose(g1.spread, g2.spread, equal_nan=True)
            and np.allclose(g1.drawdown, g2.drawdown, equal_nan=True))
def test_build_grid_matches_slow():
    
    n = 3000
    rng = np.random.default_rng(0)
    data = np.zeros((n, 9), dtype=np.float32)       # float32, like round3_params.npy
    data[:, 0] = rng.uniform(50, 500, size=n)       # balance: positive, drawdown divides by it
    data[:, 1] = rng.integers(2, 30, size=n)        # n: 2-29
    data[:, 2] = rng.integers(6, 301, size=n)       # long: 6-300
    data[:, 3] = rng.integers(2, 151, size=n)       # short: 2-150
    data[:, 4] = rng.integers(2, 30, size=n)        # bar: 2-29
    data[:, 5] = rng.integers(1, 21, size=n) * 5    # stop: 5-100 in steps of 5
    data[:, 6] = rng.integers(1, 21, size=n) * 5    # take: 5-100 in steps of 5
    data[:, 7] = rng.integers(1, 10, size=n)        # leverage: 1-9
    data[:, 8] = rng.uniform(0, 1000, size=n)       # value: below 100 and above 500

    fast = build_grid(data=data, y_col=1, x_col=4, value_col=8)
    slow = build_grid_slow(data=data, y_col=1, x_col=4, value_col=8)
    levstop_f=build_grid(data=data, y_col=7, x_col=5, value_col=8)
    levstop_s=build_grid_slow(data=data, y_col=7, x_col=5, value_col=8)
    longshort_f = build_grid(data=data, y_col=2, x_col=3, value_col=8)
    longshort_s = build_grid_slow(data=data, y_col=2, x_col=3, value_col=8)
     
    assert same(fast, slow), "Build grid result on  n x bar is different from old version"
    assert same(levstop_f, levstop_s), "Build grid result on leverage x stop is different from old version"
    assert same(longshort_f, longshort_s), "Build grid result on long x short is different from old version"
if __name__ == "__main__":
    test_build_grid_matches_slow()
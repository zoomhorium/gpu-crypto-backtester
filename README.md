# gpu-crypto-backtester

A backtester for strategies on BTC/USDT. It uses a CUDA kernel that brute-forces SMA-crossover/ADX parameter combinations.

## What it does

To reduce overfitting, the search runs as a walk-forward in three steps: 2018–20, 2020–22 and 2022–24. Each step tests only the survivors of the step before.

## Results

The search finds millions of survivors, but the top 1,000 from the third step, ranked by balance, still fail on 2024–25. This means they are still overfitted.

## How the engine is checked

The GPU kernel's results are checked against a CPU simulation, and the two agree. A bug that made ADX 14 times too small was found and fixed.

## The heatmap tool

For one pair of parameters, the tool draws six heatmaps of the survivors in each cell: median score, count, capped sum, P90, spread and drawdown. The aim is to find stable regions instead of single lucky winners.

## How to run it

You need an NVIDIA GPU and Python 3.14. Install the packages:

```
python -m pip install -r requirements.txt
```

Then, in this folder:

- `python main.py` downloads BTC/USDT 15-minute candles from Binance, runs the three search rounds on the GPU, saves the survivors in `data/` and checks them on 2024–25. This takes hours and a lot of disk space.
- `python ui.py` does the same, with parameter ranges you enter in a window.
- `python graph.py` draws the six heatmaps from `data/round3_params.npy`.
- `python -m pytest` runs the tests.

No API key is needed: the candles come from Binance's public API.

## Next

A full UI and new strategies.
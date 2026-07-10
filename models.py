from dataclasses import dataclass
import numpy as np
import cupy as cp
from typing import Iterable

@dataclass
class Ranges:
    leverage: Iterable
    stop: Iterable
    take: Iterable
    n: Iterable
    long_window: Iterable
    bar: Iterable


@dataclass
class MarketData:
    prices_close: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    volume: np.ndarray
    open_times: np.ndarray
    T: int


@dataclass
class DataForKernelGPU:
    prices: cp.ndarray
    highs: cp.ndarray
    lows: cp.ndarray
    adx_matrix: cp.ndarray
    T: int




@dataclass
class Signals:
    matrix: cp.ndarray
    reverse_map: dict
    map_of_signals: dict



@dataclass
class BestStrategyData:
    best_n: int
    best_long: int
    best_short: int
    best_bar: int
    best_stop: int
    best_take: int
    best_leverage: int
    best_balance: float
    best_score: float

@dataclass
class SurvivorsData:
    
    survivors: np.ndarray
@dataclass
class IndicatorData:
    adx: dict[int, np.ndarray]
    division: np.ndarray
@dataclass
class TrainingOutput:
    market_data: MarketData
    best_strategy: BestStrategyData
    survivors: SurvivorsData
    indicators: IndicatorData
    signals: Signals
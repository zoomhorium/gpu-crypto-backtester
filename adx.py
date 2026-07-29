
import cupy as cp
import numpy as np
from scipy.signal import lfilter, lfiltic
from models import DataForKernelGPU
from models import IndicatorData

def prepare_gpu_data(range_n,pricesClose,highs,lows):
    pricesClose_gpu = cp.array(pricesClose, dtype=cp.float32)
    highs_gpu  = cp.array(highs,  dtype=cp.float32)
    lows_gpu   = cp.array(lows,   dtype=cp.float32)
    
    
   

    # ── ADX ───────────────────────────────────────────────────────────
    plus_dm  = cp.maximum((highs_gpu[1:] - highs_gpu[:-1]), 0)
    minus_dm = cp.maximum((lows_gpu[:-1]  - lows_gpu[1:]),  0)
    tr = cp.maximum(highs_gpu[1:] - lows_gpu[1:], cp.maximum(
        abs(highs_gpu[1:] - pricesClose_gpu[:-1]),
        abs(lows_gpu[1:]  - pricesClose_gpu[:-1])
    ))
    tr_cpu       = tr.get().astype(np.float64)
    plus_dm_cpu  = plus_dm.get().astype(np.float64)
    minus_dm_cpu = minus_dm.get().astype(np.float64)
    L = len(tr_cpu)

    adx = {}
    div={}
    for n in range_n:
        b = np.array([1/n])
        a = np.array([1, -(n-1)/n])
        init_TR  = np.sum(tr_cpu[:max(n-1,1)])
        zi_TR    = lfiltic(b, a, [init_TR])
        sTR      = np.zeros(L); sTR[max(n-2,0)] = init_TR
        sTR[n-1:], _ = lfilter(b, a, tr_cpu[n-1:], zi=zi_TR)
        init_pdm = np.sum(plus_dm_cpu[:max(n-1,1)])
        zi_pdm   = lfiltic(b, a, [init_pdm])
        spdm     = np.zeros(L); spdm[max(n-2,0)] = init_pdm
        spdm[n-1:], _ = lfilter(b, a, plus_dm_cpu[n-1:], zi=zi_pdm)
        init_mdm = np.sum(minus_dm_cpu[:max(n-1,1)])
        zi_mdm   = lfiltic(b, a, [init_mdm])
        smdm     = np.zeros(L); smdm[max(n-2,0)] = init_mdm
        smdm[n-1:], _ = lfilter(b, a, minus_dm_cpu[n-1:], zi=zi_mdm)
        plus_di  = np.divide(spdm, sTR, out=np.zeros(L), where=sTR!=0) * 100
        minus_di = np.divide(smdm, sTR, out=np.zeros(L), where=sTR!=0) * 100
        di_sum   = plus_di + minus_di
        dx       = np.nan_to_num(np.where(di_sum>0, abs(plus_di-minus_di)/di_sum*100, 0.0))
        adx_arr  = np.zeros(L); adx_arr[n-1] = dx[n-1]
        zi_adx   = lfiltic(b, a, [adx_arr[n-1]])
        adx_arr[n:], _ = lfilter(b, a, dx[n:], zi=zi_adx)
        div[n] = np.divide(plus_di, minus_di, out=np.zeros(L), where=minus_di!=0)
        adx[n]   = np.concatenate([[0.0], adx_arr])

    adx_matrix = cp.array(np.stack([adx[n] for n in range_n], axis=0), dtype=cp.float32)
    T = len(pricesClose_gpu)
    
    return DataForKernelGPU(prices=pricesClose_gpu,highs=highs_gpu,lows=lows_gpu,adx_matrix=adx_matrix,T=T  ),IndicatorData(adx=adx,division=None)
def prepare_ai_data(window,pricesClose,highs,lows):
    pricesClose_gpu = cp.array(pricesClose, dtype=cp.float32)
    highs_gpu  = cp.array(highs,  dtype=cp.float32)
    lows_gpu   = cp.array(lows,   dtype=cp.float32)
    
    
   

    # ── ADX ───────────────────────────────────────────────────────────
    plus_dm  = cp.maximum((highs_gpu[1:] - highs_gpu[:-1]), 0)
    minus_dm = cp.maximum((lows_gpu[:-1]  - lows_gpu[1:]),  0)
    tr = cp.maximum(highs_gpu[1:] - lows_gpu[1:], cp.maximum(
        abs(highs_gpu[1:] - pricesClose_gpu[:-1]),
        abs(lows_gpu[1:]  - pricesClose_gpu[:-1])
    ))
    tr_cpu       = tr.get().astype(np.float64)
    plus_dm_cpu  = plus_dm.get().astype(np.float64)
    minus_dm_cpu = minus_dm.get().astype(np.float64)
    L = len(tr_cpu)

    adx = {}
    div={}
    n= window
    b = np.array([1/n])
    a = np.array([1, -(n-1)/n])
    init_TR  = np.sum(tr_cpu[:max(n-1,1)])
    zi_TR    = lfiltic(b, a, [init_TR])
    sTR      = np.zeros(L); sTR[max(n-2,0)] = init_TR
    sTR[n-1:], _ = lfilter(b, a, tr_cpu[n-1:], zi=zi_TR)
    init_pdm = np.sum(plus_dm_cpu[:max(n-1,1)])
    zi_pdm   = lfiltic(b, a, [init_pdm])
    spdm     = np.zeros(L); spdm[max(n-2,0)] = init_pdm
    spdm[n-1:], _ = lfilter(b, a, plus_dm_cpu[n-1:], zi=zi_pdm)
    init_mdm = np.sum(minus_dm_cpu[:max(n-1,1)])
    zi_mdm   = lfiltic(b, a, [init_mdm])
    smdm     = np.zeros(L); smdm[max(n-2,0)] = init_mdm
    smdm[n-1:], _ = lfilter(b, a, minus_dm_cpu[n-1:], zi=zi_mdm)
    plus_di  = np.divide(spdm, sTR, out=np.full(L, np.nan), where=sTR!=0) * 100
    minus_di = np.divide(smdm, sTR, out=np.full(L, np.nan), where=sTR!=0) * 100
    di_sum   = plus_di + minus_di
    dx       = np.nan_to_num(np.where(di_sum>0, abs(plus_di-minus_di)/di_sum*100, 0.0))
    adx_arr  = np.full(L, np.nan); adx_arr[n-1] = dx[n-1]
    zi_adx   = lfiltic(b, a, [adx_arr[n-1]])
    adx_arr[n:], _ = lfilter(b, a, dx[n:], zi=zi_adx)
    div[n] = np.concatenate([[np.nan], np.divide(plus_di, minus_di, out=np.full(L, np.nan), where=minus_di!=0)])
    adx[n]   = np.concatenate([[np.nan], adx_arr])    
    return IndicatorData(adx=adx,division=div)   
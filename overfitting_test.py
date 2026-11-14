import numpy as np
from models import Ranges
import cupy as cp
import time
import os
from paths import DATA_DIR, BASE_DIR ,ROUND3_PATH
from data import filtered_combo_generator
from saving_survivors import save
from train import kernel_loop,prepare_data
def save_train_data(kernel,round3_params,ranges):


        t0 = time.time()




        train_data=kernel_loop(2024,2025,t0,kernel,1750000,50,filtered_combo_generator,round3_params,ranges)

        save(DATA_DIR / "overfit_train_24-25.npy", train_data.survivors.survivors,ranges)
def save_random_data(seed,kernel,round3_params,ranges,length,  signals, ):

        t0 = time.time()


        rng = np.random.default_rng(seed)
        random_params = np.zeros_like(round3_params)
        pair_count = len(signals.reverse_map)
        pairs = np.array([signals.reverse_map[i] for i in range(pair_count)], dtype=np.float32)
        random_idx = rng.integers(0, pair_count, size=length)
        random_params[:,1]=rng.choice(np.unique(round3_params[:,1]),length)
        random_params[:, 2] = pairs[random_idx, 0]
        random_params[:, 3] = pairs[random_idx, 1]
        random_params[:,4]=rng.choice(np.unique(round3_params[:,4]),length)
        random_params[:,5]=rng.choice(np.unique(round3_params[:,5]),length)
        random_params[:,6]=rng.choice(np.unique(round3_params[:,6]),length)
        random_params[:,7]=rng.choice(np.unique(round3_params[:,7]),length)
        random_data=kernel_loop(2024,2025,t0,kernel,1750000,50,filtered_combo_generator,random_params,ranges)
        save(DATA_DIR / f"overfit_random_{seed}_24-25.npy", random_data.survivors.survivors,ranges)
def printout(overfit_trained,seeds):
            for i in seeds:

                overfit_random = np.load(DATA_DIR / f"overfit_random_{i}_24-25.npy")

                overfit_trained100 = overfit_trained[overfit_trained[:,0]>100]
                overfit_random100 = overfit_random[overfit_random[:,0]>100]
                print(f"Filter 100 :  Number of trained:{len(overfit_trained100)} , Number of random:{len(overfit_random100)}")

                overfit_trained150 = overfit_trained[overfit_trained[:,0]>150]
                overfit_random150 = overfit_random[overfit_random[:,0]>150]
                print(f"Filter 150 :  Number of trained:{len(overfit_trained150)} , Number of random:{len(overfit_random150)}")

                overfit_trained170 = overfit_trained[overfit_trained[:,0]>170]
                overfit_random170 = overfit_random[overfit_random[:,0]>170]
                print(f"Filter 170 :  Number of trained:{len(overfit_trained170)} , Number of random:{len(overfit_random170)}")

                overfit_trained_eq100 = overfit_trained[overfit_trained[:,0]==100]
                overfit_random_eq100 = overfit_random[overfit_random[:,0]==100]
                print(f"Filter 100 equal :  Number of trained:{len(overfit_trained_eq100)} , Number of random:{len(overfit_random_eq100)}")
def preparation():
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

        t0 = time.time()
        round3_params = np.load(ROUND3_PATH)
        ranges=Ranges(
            leverage=range(int(np.min(np.unique(round3_params[:,7]))),int(np.max(np.unique(round3_params[:,7]))+1)),
            stop=range(int(np.min(np.unique(round3_params[:,5]))),int(np.max(np.unique(round3_params[:,5]))+1),5),
            take=range(int(np.min(np.unique(round3_params[:,6]))),int(np.max(np.unique(round3_params[:,6]))+1),5),
            n=range(4,30),##the round files’ n labels are 2 too small, so labels 4–27 mean periods 6–29
            long_window=range(int(np.min(np.unique(round3_params[:,2]))),int(np.max(np.unique(round3_params[:,2]))+1)),
            bar=range(int(np.min(np.unique(round3_params[:,4]))),int(np.max(np.unique(round3_params[:,4]))+1))
        )

        length=len(round3_params)
        cpu_data, gpu_data, signals, adx=prepare_data(2024,2025,t0,ranges)
        return kernel,round3_params,ranges,length,  signals
if __name__ == "__main__":
    kernel,round3_params,ranges,length,  signals=preparation()
    save_train_data(kernel,round3_params,ranges)
    seeds=range(0,6)
    for i in seeds:

        save_random_data(i,kernel,round3_params,ranges,length,  signals)
    overfit_trained = np.load(DATA_DIR / "overfit_train_24-25.npy")
    printout(overfit_trained,seeds)




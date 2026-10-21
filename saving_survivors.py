import numpy as np
from pathlib import Path    
import dataclasses
import json
from models import Ranges
def save(path,survivors,ranges):
    np.save(path, survivors)
    side_path=path.with_suffix(".json")
    ranges_dict=dataclasses.asdict(ranges)
    
    with open(side_path, "w") as f:

        json.dump(ranges_dict, f, indent=2,default=rangeJson)
def rangeJson(obj):
    if isinstance(obj, range):
        start_val = obj.start
        stop_val  = obj.stop
        step_val  = obj.step
        return start_val,stop_val ,step_val
    raise TypeError(f"Can't write {type(obj).__name__} to JSON")
if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent
    survivors=[1,12,2,2,3,23,435,31,242,14,12,41,24,124,12,41]
    path=DATA_DIR / "scratch_test.npy"
    ranges=Ranges(range(1,10),range(1,10),range(1,10),range(1,10),range(1,10),range(1,10))
    save(path,survivors,ranges)
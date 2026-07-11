from train import stack_survivors
from models import NoSurvivorsError
import pytest
import numpy as np

def test_stack_survivors_empty_raises():
    with pytest.raises(NoSurvivorsError):
        stack_survivors([], 170, 2018, 2020)
def test_stack_survivors_stacks():
    best_params_list = []
    chunk_data = np.column_stack([
                        200,
                        
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10
                    ])
    best_params_list.append(chunk_data)
    best_params_list.append(chunk_data)
    best_params_list.append(chunk_data)
    best_params_list.append(chunk_data)
    best_params = stack_survivors(best_params_list, 170, 2018, 2020)
    assert best_params.shape == (4, 9),        "WRONG SHAPE"
if __name__ == "__main__":
    test_stack_survivors_empty_raises()
    test_stack_survivors_stacks()
    print("TESTS PASSED")
import numpy as np
from paths import ROUND3_PATH

array   =   np.load(ROUND3_PATH)
print(array.shape)
print(array.size)
print(array.dtype.names)
print(array[:3 ])
print(array.dtype)


print(np.unique(array[:,4]))
print(np.percentile(array[:,8],99))
print(np.percentile(array[:,8],50))
print(np.max(array[:,8]))
print(np.min(array[:, 8]))
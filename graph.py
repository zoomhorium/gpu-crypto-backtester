from paths import DATA_DIR, BASE_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
import matplotlib
def build_grid(data, y_col, x_col, value_col, y_values=None, x_values=None):

    if y_values is None:
        y_values=range(int(np.min(np.unique(data[:,y_col]))),int(np.max(np.unique(data[:,y_col])))+1 )
    if x_values is None:
        x_values=range(int(np.min(np.unique(data[:,x_col]))),int(np.max(np.unique(data[:,x_col])))+1)
    medians = np.full(( len(y_values),len(x_values)), np.nan)
    counts  = np.zeros(( len(y_values),len(x_values)), dtype=int)
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            mask = (data[:, y_col] == y) & (data[:, x_col] == x)
            counts[i, j] = mask.sum()
            if counts[i, j]>=100:
                medians[i, j] = np.median(data[mask, value_col])
            else:
                medians[i, j] = np.nan
        

    return y_values, x_values, medians, counts

def draw_grid_pair(axA, axB, y_values, x_values, medians, counts, y_label, x_label,cmap_a,cmap_b):
    im_a = axA.imshow(medians, cmap=cmap_a,
                  norm=TwoSlopeNorm(vcenter=100,
                                    vmin=np.nanmin(medians),
                                    vmax=np.nanmax(medians)))
    im_b = axB.imshow(counts,   cmap=cmap_b, norm=LogNorm(vmin=10, vmax=counts.max()))
    axA.invert_yaxis()  
    axA.set_xlabel(x_label)
    axA.set_ylabel(y_label)
    axB.set_xlabel(x_label)
    axB.set_ylabel(y_label)
    axB.invert_yaxis()  
    axA.figure.colorbar(im_a, ax=axA)
    axB.figure.colorbar(im_b, ax=axB)
    axA.set_xticks(range(len(x_values)), list(x_values))
    axA.set_yticks(range(len(y_values)), list(y_values))
    axB.set_xticks(range(len(x_values)), list(x_values))
    axB.set_yticks(range(len(y_values)), list(y_values))    

cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6",under="0.8")

a = np.load(DATA_DIR / "round3_params.npy")
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(25, 25))
y_values,x_values,medians,counts = build_grid(data=a,y_col=1,x_col=4,value_col=8)

draw_grid_pair(axA, axB, y_values, x_values, medians, counts, "n", "bar",cmap_a,cmap_b)




            
for i, y in enumerate(y_values):
    j = np.nanargmin(medians[i, :])
    print(x_values[j], counts[i, j], y)
print(np.min(a[:,8]),np.min(a[:,0]))



fig2.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)


plt.show()
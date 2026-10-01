from paths import DATA_DIR, BASE_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib
def build_grid(data, y_col, x_col, value_col, y_values=None, x_values=None, lower_limit=100, upper_limit=500):

    if y_values is None:
        y_values=range(int(np.min(np.unique(data[:,y_col]))),int(np.max(np.unique(data[:,y_col])))+1 )
    if x_values is None:
        x_values=range(int(np.min(np.unique(data[:,x_col]))),int(np.max(np.unique(data[:,x_col])))+1)
    medians = np.full(( len(y_values),len(x_values)), np.nan)
    counts  = np.zeros(( len(y_values),len(x_values)), dtype=int)
    capped_sum  = np.zeros(( len(y_values),len(x_values)), dtype=float)
    p10 = np.full(( len(y_values),len(x_values)), np.nan) 
    p90 = np.full(( len(y_values),len(x_values)), np.nan)
    spread = np.zeros(( len(y_values),len(x_values)), dtype=float)
    
    for i, y in enumerate(y_values):

        for j, x in enumerate(x_values):
            mask = (data[:, y_col] == y) & (data[:, x_col] == x)
            counts[i, j] = mask.sum()
            if counts[i, j]:
                medians[i, j] = np.median(data[mask, value_col])

                p10[i, j],p90[i, j] = np.percentile(data[mask, value_col],[10,90])
                spread[i,j] = p90[i, j]- p10[i, j]
                norm_data = (data[mask, value_col] - lower_limit) / (upper_limit - lower_limit)


                capped_sum[i,j] = np.sum(np.clip(norm_data, 0, 1))
        

    return y_values, x_values, medians, counts , capped_sum ,p10 ,p90 , spread

def draw_grid_pair(axA, axB, axC, axD, axE, y_values, x_values, medians, counts,capped_sum,p10 ,p90 , spread, y_label, x_label,cmap_a,cmap_b,cmap_c,cmap_e,cmap_d,vcenter=100, thin_below=100):
    im_a = axA.imshow(medians, cmap=cmap_a,
                  norm=TwoSlopeNorm(vcenter=vcenter,
                                    vmin=np.nanmin(medians),
                                    vmax=np.nanmax(medians)))
    im_b = axB.imshow(counts,   cmap=cmap_b, norm=LogNorm(vmin=10, vmax=counts.max()))
   
    im_c = axC.imshow(capped_sum, cmap=cmap_c,norm=LogNorm(vmin=1, vmax=capped_sum.max()))
    im_d = axD.imshow(p90, cmap=cmap_d,
                      norm=TwoSlopeNorm(vcenter=250,
                                        vmin=np.nanmin(p90),
                                        vmax=np.nanmax(p90)))
    im_e = axE.imshow(spread, cmap=cmap_e,norm=LogNorm(vmin=1, vmax=spread.max()))
    axA.invert_yaxis()  

    axA.set_xlabel(x_label)
    axA.set_ylabel(y_label)
    axB.set_xlabel(x_label)
    axB.set_ylabel(y_label)
    axB.invert_yaxis()  
    axC.set_xlabel(x_label)
    axC.set_ylabel(y_label)
    axC.invert_yaxis()
    axD.set_xlabel(x_label)
    axD.set_ylabel(y_label)
    axD.invert_yaxis()
    axE.set_xlabel(x_label)
    axE.set_ylabel(y_label)
    axE.invert_yaxis()
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            if  thin_below>counts[i,j]>0: 
                axA.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="k", lw=1.5))
    axA.figure.colorbar(im_a, ax=axA)
    axB.figure.colorbar(im_b, ax=axB)
    axC.figure.colorbar(im_c, ax=axC)
    axD.figure.colorbar(im_d, ax=axD)
    axE.figure.colorbar(im_e, ax=axE)
    axA.set_xticks(range(len(x_values)), list(x_values))
    axA.set_yticks(range(len(y_values)), list(y_values))
    axB.set_xticks(range(len(x_values)), list(x_values))
    axB.set_yticks(range(len(y_values)), list(y_values))   
    axC.set_xticks(range(len(x_values)), list(x_values))
    axC.set_yticks(range(len(y_values)), list(y_values))
    axD.set_xticks(range(len(x_values)), list(x_values))
    axD.set_yticks(range(len(y_values)), list(y_values))  
    axE.set_xticks(range(len(x_values)), list(x_values))
    axE.set_yticks(range(len(y_values)), list(y_values))     

cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6",under="0.8")
cmap_c = matplotlib.colormaps["plasma"].with_extremes(bad="0.6",under="0.8")
cmap_d = matplotlib.colormaps["PuBu"].with_extremes(bad="0.6")
cmap_e = matplotlib.colormaps["magma"].with_extremes(bad="0.6",under="0.8")

a = np.load(DATA_DIR / "round3_params.npy")
fig2, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(25, 25))
y_values,x_values,medians,counts,capped_sum,p10 ,p90 , spread = build_grid(data=a,y_col=1,x_col=4,value_col=8)

draw_grid_pair(axA, axB, axC, axD, axE,  y_values, x_values, medians, counts, capped_sum,p10 ,p90 , spread, "n", "bar",cmap_a,cmap_b,cmap_c,cmap_e,cmap_d )




for i, y in enumerate(y_values):
    row = np.where(counts[i] > 99, medians[i], np.nan)
    j = np.nanargmin(row)
    print(x_values[j], counts[i, j], y)
print(np.min(a[:,8]),np.min(a[:,0]))

print(np.nanmin(medians), np.nanmax(medians))

thin = (counts > 0) & (counts < 100)
print(thin.sum())
print(np.nanmin(p90))
 
fig2.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)


plt.show()
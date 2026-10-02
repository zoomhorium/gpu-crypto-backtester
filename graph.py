from paths import DATA_DIR, BASE_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib
from models import Grid


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
                spread[i,j] =( p90[i, j]- p10[i, j])/medians[i, j]
                norm_data = (data[mask, value_col] - lower_limit) / (upper_limit - lower_limit)


                capped_sum[i,j] = np.sum(np.clip(norm_data, 0, 1))
        

    return Grid(y_values=y_values, x_values=x_values, medians=medians, counts=counts , capped_sum=capped_sum ,p10=p10 ,p90=p90 , spread=spread)

def draw_grid_pair(axA, axB, axC, axD, axE, grid, y_label, x_label,cmap_a,cmap_b,cmap_c,cmap_e,cmap_d,vcenter=100, thin_below=100):
    normA=TwoSlopeNorm(vcenter=vcenter,
                                        vmin=np.nanmin(grid.medians),
                                        vmax=np.nanmax(grid.medians))
    draw_panel(ax=axA,values=grid.medians,cmap=cmap_a,norm=normA,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,title="MEDIAN")
    normB=LogNorm(vmin=10, vmax=grid.counts.max())
    draw_panel(ax=axB,values=grid.counts,cmap=cmap_b,norm=normB,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,title="COUNT")
    normC=LogNorm(vmin=1, vmax=grid.capped_sum.max())
    draw_panel(ax=axC,values=grid.capped_sum,cmap=cmap_c,norm=normC,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,title="CAPPED SUM")
    
    normD=TwoSlopeNorm(vcenter=100,
                                            vmin=np.nanmin(grid.p90),
                                            vmax=np.nanmax(grid.p90))
    draw_panel(ax=axD,values=grid.p90,cmap=cmap_d,norm=normD,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,title="P90")
    
    normE=LogNorm(vmin=0.1, vmax=grid.spread.max())
    draw_panel(ax=axE,values=grid.spread,cmap=cmap_e,norm=normE,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,title="SPREAD")
    
    

   
    draw_patches(ax=axA,counts=grid.counts,y_values=grid.y_values,x_values=grid.x_values,threshold=thin_below)


def draw_panel(ax, values, cmap, norm, y_values, x_values, y_label, x_label,title):
    im = ax.imshow(values,   cmap=cmap, norm=norm)
    ax.invert_yaxis()  
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(x_values)), list(x_values))
    ax.set_yticks(range(len(y_values)), list(y_values))
    ax.set_title(title)
def draw_patches(ax, counts, y_values, x_values,threshold ):
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            if  threshold>counts[i,j]>0: 
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="k", lw=1.5))
cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6",under="0.8")
cmap_c = matplotlib.colormaps["plasma"].with_extremes(bad="0.6",under="0.8")
cmap_d = matplotlib.colormaps["PuBu"].with_extremes(bad="0.6")
cmap_e = matplotlib.colormaps["magma"].with_extremes(bad="0.6",under="0.8")

a = np.load(DATA_DIR / "round3_params.npy")
fig2, (axA, axB, axC, axD, axE) = plt.subplots(1, 5, figsize=(25, 25))
grid = build_grid(data=a,y_col=1,x_col=4,value_col=8)

draw_grid_pair(axA, axB, axC, axD, axE,  grid, "n", "bar",cmap_a,cmap_b,cmap_c,cmap_e,cmap_d )
    



for i, y in enumerate(grid.y_values):
    row = np.where(grid.counts[i] > 99, grid.medians[i], np.nan)
    j = np.nanargmin(row)
    print(grid.x_values[j], grid.counts[i, j], y)
print(np.min(a[:,8]),np.min(a[:,0]))

print(np.nanmin(grid.medians), np.nanmax(grid.medians))

thin = (grid.counts > 0) & (grid.counts < 100)
print(thin.sum())
print(np.nanmin(grid.p90))
ok = grid.counts > 99
print(np.corrcoef(np.log(grid.counts[ok]), np.log(grid.spread[ok]))[0, 1])
fig2.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)
A = plt.imread(BASE_DIR / "before.png")
B = plt.imread(BASE_DIR / "bar_n_heatmap.png")
print(np.array_equal(A, B))

plt.show()
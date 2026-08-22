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

cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6")

a = np.load(DATA_DIR / "round3_params.npy")
b = np.load(DATA_DIR / "round2_params.npy")

fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 26))

print(len(np.unique(a[:, 2])),len(np.unique(a[:, 3])))
print(np.unique(a[:, 2]))
print(np.unique(a[:, 3]))
ax1.scatter(a[:, 4], a[:, 8], s=2, alpha=0.05, linewidths=0,label="scores")
bars = np.unique(a[:, 4])
ns   = np.unique(a[:, 1])
y_values,x_values,medians,counts = build_grid(data=a,y_col=1,x_col=4,value_col=8)



            
for i, y in enumerate(y_values):
    j = np.nanargmin(medians[i, :])
    print(x_values[j], counts[i, j], y)
print(np.min(a[:,8]),np.min(a[:,0]))
print(bars)
print(ns)
print(medians.shape, counts.min(), counts.max())
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(25, 25))
im_a = axA.imshow(medians, cmap=cmap_a,
                  norm=TwoSlopeNorm(vcenter=100,
                                    vmin=np.nanmin(medians),
                                    vmax=np.nanmax(medians)))
im_b = axB.imshow(counts,   cmap=cmap_b, norm=LogNorm(vmin=1, vmax=counts.max()))
axA.invert_yaxis()  
axA.set_xlabel("bar")
axA.set_ylabel("n")
axB.invert_yaxis()  
fig2.colorbar(im_a, ax=axA)
fig2.colorbar(im_b, ax=axB)

axA.set_xticks(range(len(x_values)), list(x_values))
axA.set_yticks(range(len(y_values)), list(y_values))
axB.set_xticks(range(len(x_values)), list(x_values))
axB.set_yticks(range(len(y_values)), list(y_values))
fig2.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)
median=[]
for bar in bars:
    idx = np.where(bar==a[:, 4])[0]
    median.append(np.median(a[idx,8]))

ax1.plot(bars,median,marker="o",label="median",color="red")
ax2.scatter(a[:, 4], (1 - a[:, 8] / a[:, 0])*100, s=2, alpha=0.05, linewidths=0,label="drawdown",color="green")
ax1.set_yscale("log")
ax1.set_xlabel("bar (ADX threshold)")
ax1.set_ylabel("score")
ax2.set_xlabel("bar (ADX threshold)")
ax2.set_ylabel("drawdown %")

ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)
ax1.legend( )
ax2.legend( )


hb = ax3.hexbin(b[:, 4], b[:, 8], yscale="log", bins="log",
                gridsize=(26, 40), cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax3)
bars=np.unique(b[:, 4])
print(np.min(b[:,8]),np.min(b[:,0]))
median=[]
for bar in bars:
    idx = np.where(bar==b[:, 4])[0]
    median.append(np.median(b[idx,8]))

ax3.plot(bars,median,marker="o",label="median",color="red")
hb2 = ax4.hexbin(b[:, 4], (1 - b[:, 8] / b[:, 0])*100,  bins="log",
                gridsize=(26, 16), cmap="viridis", mincnt=1)
fig.colorbar(hb2, ax=ax4)
ax3.set_yscale("log")
ax3.set_xlabel("bar (ADX threshold)")
ax3.set_ylabel("score")
ax4.set_xlabel("bar (ADX threshold)")
ax4.set_ylabel("drawdown %")

ax3.grid(True, alpha=0.3)
ax4.grid(True, alpha=0.3)
ax3.legend( )
ax1.set_title("round3 — bar vs score")
ax2.set_title("round3 — bar vs drawdown")
ax3.set_title("round2 — bar vs score")
ax4.set_title("round2 — bar vs drawdown")

fig.tight_layout()
fig.savefig(BASE_DIR / "bar_vs_score.png", dpi=150)

plt.show()
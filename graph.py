from paths import DATA_DIR, BASE_DIR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
import matplotlib
cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6")

a = np.load(DATA_DIR / "round3_params.npy")
b = np.load(DATA_DIR / "round2_params.npy")
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 8))


ax1.scatter(a[:, 4], a[:, 8], s=2, alpha=0.05, linewidths=0,label="scores")
bars = np.unique(a[:, 4])
ns   = np.unique(a[:, 1])

medians = np.full((len(ns), len(bars)), np.nan)
counts  = np.zeros((len(ns), len(bars)), dtype=int)

for i, n in enumerate(ns):
    for j, bar in enumerate(bars):
        mask = (a[:, 1] == n) & (a[:, 4] == bar)
        counts[i, j] = mask.sum()
        if counts[i, j]:
            medians[i, j] = np.median(a[mask, 8])

print(np.min(a[:,8]),np.min(a[:,0]))
print(bars)
print(ns)
print(medians.shape, counts.min(), counts.max())
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(25, 12))
im_a = axA.imshow(medians, cmap=cmap_a,
                  norm=TwoSlopeNorm(vcenter=100,
                                    vmin=np.nanmin(medians),
                                    vmax=np.nanmax(medians)))
im_b = axB.imshow(counts,   cmap=cmap_b, norm=LogNorm(vmin=1, vmax=counts.max()))
axA.invert_xaxis()  
axB.invert_xaxis()  
fig2.colorbar(im_a, ax=axA)
fig2.colorbar(im_b, ax=axB)

axA.set_xticks(range(len(bars)), [int(v) for v in bars])
axA.set_yticks(range(len(ns)), [int(v) for v in ns])
axB.set_xticks(range(len(bars)), [int(v) for v in bars])
axB.set_yticks(range(len(ns)), [int(v) for v in ns])
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
ax1.set_xlabel("bar (ADX threshold)")
ax1.set_ylabel("drawdown")
ax1.set_title("round3 — bar vs drawdown")
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
                gridsize=(26, 40), cmap="viridis", mincnt=1)
fig.colorbar(hb2, ax=ax4)
ax3.set_yscale("log")
ax3.set_xlabel("bar (ADX threshold)")
ax3.set_ylabel("score")
ax3.set_xlabel("bar (ADX threshold)")
ax3.set_ylabel("drawdown")
ax3.set_title("round2 — bar vs drawdown") 
ax3.grid(True, alpha=0.3)
ax4.grid(True, alpha=0.3)
ax3.legend( )

fig.tight_layout()
fig.savefig(BASE_DIR / "bar_vs_score.png", dpi=150)
plt.show()
from paths import  BASE_DIR ,ROUND3_PATH
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import matplotlib

from models import Grid
from matplotlib.colors import Colormap, Normalize
from dataclasses import dataclass

import math
from numbers import Number


class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, axes):
        self.axes = list(axes.flat)
        self.backgrounds = []
        self.horizontal_lines=[]
        self.vertical_lines = []

        for ax in self.axes:
            self.horizontal_lines.append(ax.axhline(color='k', lw=0.8, ls='--',animated=True))
            self.vertical_lines.append(ax.axvline(color='k', lw=0.8, ls='--',animated=True))
            self.set_cross_hair_visible(False)
            self.backgrounds.append(None)

        self.axes[0].figure.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        for hl,vl in zip(self.horizontal_lines,self.vertical_lines,strict=True):
            hl.set_visible(visible)
            vl.set_visible(visible)

    def create_new_background(self):
        new_background=[]
        for ax in self.axes:
            new_background.append(ax.figure.canvas.copy_from_bbox(ax.bbox))
        self.backgrounds=new_background

    def on_mouse_move(self, event):

        for ax,hl,vl,background in zip(self.axes,self.horizontal_lines,self.vertical_lines,self.backgrounds,strict=True):
            if background is None:
                self.create_new_background()

            if event.inaxes not in self.axes:
                self.set_cross_hair_visible(False)
                ax.figure.canvas.restore_region(background)

            else:
                self.set_cross_hair_visible(True)
                # update the line positions
                x, y = event.xdata, event.ydata
                hl.set_ydata([y])
                vl.set_xdata([x])
                ax.figure.canvas.restore_region(background)
                ax.draw_artist(hl)
                ax.draw_artist(vl)
        for ax in self.axes:
            ax.figure.canvas.blit(ax.bbox)

@dataclass
class Panel:
    values: np.ndarray
    cmap: Colormap
    norm: Normalize
    title : str
    mark_thin: bool = False
def get_range(column):
    uniq=np.unique(column)
    step=np.gcd.reduce(np.diff(uniq).astype(np.int32))
    if step==0:
        step=1
    new_range=range(int(np.min(uniq)),int(np.max(uniq))+1,step)
    return new_range
def build_grid(data, y_col, x_col, value_col,  lower_limit=100, upper_limit=500,balance_col=0):

    y_values=get_range(data[:,y_col])
    x_values=get_range(data[:,x_col])
    medians = np.full(( len(y_values),len(x_values)), np.nan)
    counts  = np.zeros(( len(y_values),len(x_values)), dtype=int)
    capped_sum  = np.zeros(( len(y_values),len(x_values)), dtype=float)
    p10 = np.full(( len(y_values),len(x_values)), np.nan)
    p90 = np.full(( len(y_values),len(x_values)), np.nan)
    spread = np.zeros(( len(y_values),len(x_values)), dtype=float)
    drawdown = np.full(( len(y_values),len(x_values)), np.nan)
    columns = len(x_values)
    cells= (data[:, y_col] -y_values.start)/y_values.step*columns +(data[:, x_col]-x_values.start)/x_values.step
    cells= cells.astype(int)
    order  = np.argsort(cells)
    counts_new = np.bincount(cells, minlength=len(y_values) * len(x_values))
    ends = np.cumsum(counts_new)
    starts = ends - counts_new
    sorted_value=data[order, value_col]
    sorted_balance=data[order, balance_col]
    for i, y in enumerate(y_values):

        for j, x in enumerate(x_values):

            c = i * columns + j
            values   = sorted_value[starts[c]:ends[c]]
            balances = sorted_balance[starts[c]:ends[c]]
            counts[i, j] = len(values)

            if counts[i, j]:
                medians[i, j] = np.median(values)
                drawdown[i,j] = np.median(1 - values / balances)
                p10[i, j],p90[i, j] = np.percentile(values,[10,90])
                spread[i,j] =( p90[i, j]- p10[i, j])/medians[i, j]
                norm_data = (values - lower_limit) / (upper_limit - lower_limit)


                capped_sum[i,j] = np.sum(np.clip(norm_data, 0, 1))


    return Grid(y_values=y_values, x_values=x_values, medians=medians, counts=counts , capped_sum=capped_sum ,p10=p10 ,p90=p90 , spread=spread,drawdown = drawdown)

def draw_panels(axes, grid, y_label, x_label, cmap_a, cmap_b, cmap_c, cmap_d, cmap_e, cmap_f, vcenter=100, thin_below=100):
    def format_coord_custom(x, y):
        if x<-0.5 or y<-0.5 or len(grid.y_values) - 0.5<=y or len(grid.x_values) - 0.5<=x:
            return "Too far"
        value_str = ""
        for panel in panels:
            value_str=value_str+panel.title+" = "+f"{panel.values[math.floor(y + 0.5)][math.floor(x + 0.5)].item():0.5g} "+"   "

        return f'{x_label}={grid.x_values[math.floor((x+0.5))]}, {y_label}={grid.y_values[math.floor((y+0.5))]}'+'\n' + value_str


    panels = []
    normA=TwoSlopeNorm(vcenter=vcenter,
                                        vmin=np.nanmin(grid.medians),
                                        vmax=np.nanmax(grid.medians))
    panels.append(Panel(values=grid.medians,cmap=cmap_a,norm=normA,title="MEDIAN",mark_thin=True))

    normB=LogNorm(vmin=10, vmax=grid.counts.max())
    panels.append(Panel(values=grid.counts,cmap=cmap_b,norm=normB,title="COUNT"))
    normC=LogNorm(vmin=1, vmax=grid.capped_sum.max())
    panels.append(Panel(values=grid.capped_sum,cmap=cmap_c,norm=normC,title="CAPPED SUM"))

    normD=TwoSlopeNorm(vcenter=100,
                                            vmin=np.nanmin(grid.p90),
                                            vmax=np.nanmax(grid.p90))
    panels.append(Panel(values=grid.p90,cmap=cmap_d,norm=normD,title="P90"))


    normE=LogNorm(vmin=0.1, vmax=grid.spread.max())
    panels.append(Panel(values=grid.spread,cmap=cmap_e,norm=normE,title="SPREAD"))
    normF=Normalize(vmin=0, vmax=1)
    panels.append(Panel(values=grid.drawdown,cmap=cmap_f,norm=normF,title="DRAWDOWN"))

    for ax, panel in zip(axes.flat, panels, strict=True):
        draw_panel(ax=ax,y_values=grid.y_values,x_values=grid.x_values,y_label=y_label,x_label=x_label,panel=panel,threshold=thin_below,counts=grid.counts,format_coord_custom =format_coord_custom )



def format_cursor_data_custom( data):
    try:
        data[0]
    except (TypeError, IndexError):
        data = [data]
    data_str = ', '.join(f'{item:0.5g}' for item in data
                            if isinstance(item, Number))
    return "[" + data_str + "]"
def draw_panel(ax, y_values, x_values, y_label, x_label,panel,threshold,counts,format_coord_custom ):

    im = ax.imshow(panel.values,   cmap=panel.cmap, norm=panel.norm)
    im.format_cursor_data=format_cursor_data_custom
    ax.invert_yaxis()
    ax.format_coord = format_coord_custom

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(x_values)), list(x_values))
    ax.set_yticks(range(len(y_values)), list(y_values))
    ax.tick_params(labelbottom=True,labelleft=True)
    ax.set_title(panel.title)
    if panel.mark_thin:
        draw_patches(ax=ax,counts=counts,y_values=y_values,x_values=x_values,threshold=threshold)
def draw_patches(ax, counts, y_values, x_values,threshold):
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            if  threshold>counts[i,j]>0:
                ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="k", lw=1.5))



def graph(fig,grid,y_label, x_label):
    axes=fig.subplots(2, 3, sharex="all",sharey="all")
    cursor = Cursor(axes)
    cmap_a = matplotlib.colormaps["RdBu"].with_extremes(bad="0.6")
    cmap_b = matplotlib.colormaps["viridis"].with_extremes(bad="0.6",under="0.8")
    cmap_c = matplotlib.colormaps["plasma"].with_extremes(bad="0.6",under="0.8")
    cmap_d = matplotlib.colormaps["PuBu"].with_extremes(bad="0.6")
    cmap_e = matplotlib.colormaps["magma"].with_extremes(bad="0.6",under="0.8")
    cmap_f = matplotlib.colormaps["inferno"].with_extremes(bad="0.6",under="0.8")



    draw_panels(axes, grid, y_label, x_label, cmap_a, cmap_b, cmap_c, cmap_d, cmap_e, cmap_f)
    fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
    return cursor,axes

if __name__ == "__main__":
    survivors = np.load(ROUND3_PATH)
    survivors_lev3 = survivors[survivors[:, 7] == 3]

    fig_n_bar = plt.figure(figsize=(24, 14), layout="constrained")
    fig_lev_stop = plt.figure(figsize=(24, 14), layout="constrained")
    fig_lev3 = plt.figure(figsize=(24, 14), layout="constrained")

    grid_n_bar = build_grid(data=survivors, y_col=1, x_col=4, value_col=8)
    grid_lev_stop = build_grid(data=survivors, y_col=7, x_col=5, value_col=8)
    grid_lev3 = build_grid(data=survivors_lev3, y_col=7, x_col=5, value_col=8)

    # Keep the cursors in variables, or the crosshairs stop working.
    cursor_n_bar, axes_n_bar = graph(fig=fig_n_bar, grid=grid_n_bar, y_label="n", x_label="bar")
    cursor_lev_stop, axes_lev_stop = graph(fig=fig_lev_stop, grid=grid_lev_stop, y_label="leverage", x_label="stop")
    cursor_lev3, axes_lev3 = graph(fig=fig_lev3, grid=grid_lev3, y_label="leverage", x_label="stop")

    fig_n_bar.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)

    hover_n_bar = axes_n_bar[0][0].format_coord
    hover_lev_stop = axes_lev_stop[0][0].format_coord
    hover_lev3 = axes_lev3[0][0].format_coord

    # n x bar
    first_bar, last_bar = grid_n_bar.x_values[0], grid_n_bar.x_values[-1]
    first_n, last_n = grid_n_bar.y_values[0], grid_n_bar.y_values[-1]
    last_col, last_row = len(grid_n_bar.x_values) - 1, len(grid_n_bar.y_values) - 1
    x_edge = last_col + 0.5          # where the last column ends
    y_edge = last_row + 0.5          # where the last row ends
    assert hover_n_bar(0, 0).split("\n")[0] == f"bar={first_bar}, n={first_n}", "n x bar: first cell"
    assert hover_n_bar(last_col, last_row).split("\n")[0] == f"bar={last_bar}, n={last_n}", "n x bar: last cell"
    assert f"COUNT = {grid_n_bar.counts[0, 22].item():0.5g} " in hover_n_bar(22, 0), "n x bar: hover shows the wrong count"
    assert hover_n_bar(x_edge - 0.01, 0).split("\n")[0] == f"bar={last_bar}, n={first_n}", "n x bar: last column, just inside"
    assert hover_n_bar(x_edge, 0) == "Too far", "n x bar: last column, on the edge"
    assert hover_n_bar(-0.5, 0).split("\n")[0] == f"bar={first_bar}, n={first_n}", "n x bar: first column, on the edge"
    assert hover_n_bar(-0.51, 0) == "Too far", "n x bar: first column, just outside"
    assert hover_n_bar(0, y_edge - 0.01).split("\n")[0] == f"bar={first_bar}, n={last_n}", "n x bar: last row, just inside"
    assert hover_n_bar(0, y_edge) == "Too far", "n x bar: last row, on the edge"
    assert hover_n_bar(0, -0.5).split("\n")[0] == f"bar={first_bar}, n={first_n}", "n x bar: first row, on the edge"
    assert hover_n_bar(0, -0.51) == "Too far", "n x bar: first row, just outside"

    # leverage x stop
    last_stop, first_leverage = grid_lev_stop.x_values[-1], grid_lev_stop.y_values[0]
    stop_edge = len(grid_lev_stop.x_values) - 0.5
    assert hover_lev_stop(stop_edge - 0.01, 0).split("\n")[0] == f"stop={last_stop}, leverage={first_leverage}", "leverage x stop: last column, just inside"
    assert hover_lev_stop(stop_edge, 0) == "Too far", "leverage x stop: last column, on the edge"

    # leverage = 3 only
    assert list(grid_lev3.y_values) == [3], "leverage = 3 only: the y axis must hold just 3"
    assert hover_lev3(0, 0).split("\n")[0] == f"stop={grid_lev3.x_values[0]}, leverage=3", "leverage = 3 only: first cell"

    plt.show()

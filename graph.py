from paths import DATA_DIR, BASE_DIR
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
    
def build_grid(data, y_col, x_col, value_col, y_values=None, x_values=None, lower_limit=100, upper_limit=500,balance_col=0):

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
    drawdown = np.full(( len(y_values),len(x_values)), np.nan)
    for i, y in enumerate(y_values):

        for j, x in enumerate(x_values):
            mask = (data[:, y_col] == y) & (data[:, x_col] == x)
            counts[i, j] = mask.sum()
            if counts[i, j]:
                medians[i, j] = np.median(data[mask, value_col])
                drawdown[i,j] = np.median(1 - data[mask, value_col] / data[mask, balance_col])
                p10[i, j],p90[i, j] = np.percentile(data[mask, value_col],[10,90])
                spread[i,j] =( p90[i, j]- p10[i, j])/medians[i, j]
                norm_data = (data[mask, value_col] - lower_limit) / (upper_limit - lower_limit)


                capped_sum[i,j] = np.sum(np.clip(norm_data, 0, 1))
        

    return Grid(y_values=y_values, x_values=x_values, medians=medians, counts=counts , capped_sum=capped_sum ,p10=p10 ,p90=p90 , spread=spread,drawdown = drawdown)

def draw_grid_pair(axes, grid, y_label, x_label,cmap_a,cmap_b,cmap_c,cmap_e,cmap_d,cmap_f,vcenter=100, thin_below=100):
    def format_coord_custom(x, y):
        if x<-0.5 or y<-0.5 or np.max(grid.y_values)-np.min(grid.y_values)+0.5<=y or np.max(grid.x_values)-np.min(grid.x_values)+0.5<=x:
            return "Too far"
        value_str = ""
        for panel in panels:
            value_str=value_str+panel.title+" = "+f"{panel.values[math.floor(y + 0.5)][math.floor(x + 0.5)].item():0.5g} "+"   "
        
        return f'{x_label}={math.floor((x+0.5))+np.min(np.unique(grid.x_values))}, {y_label}={math.floor((y+0.5))+np.min(np.unique(grid.y_values))}'+'\n' + value_str
    
    
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

    

    draw_grid_pair(axes,  grid, y_label, x_label,cmap_a,cmap_b,cmap_c,cmap_e,cmap_d ,cmap_f)
    fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
    return cursor,axes
if __name__ == "__main__":
    fig = plt.figure(figsize=(24, 14), layout="constrained") 

    a = np.load(DATA_DIR / "round3_params.npy")
    grid = build_grid(data=a,y_col=1,x_col=4,value_col=8)
    cursor,axes=graph( fig=fig,grid=grid,y_label= "n", x_label="bar")


    

    
    

   



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

    fig.savefig(BASE_DIR / "bar_n_heatmap.png", dpi=150)

    fc = axes[0][0].format_coord
    x_edge = len(grid.x_values) - 0.5          # where the last column ends
    y_edge = len(grid.y_values) - 0.5          # where the last row ends
    first_bar, last_bar = grid.x_values[0], grid.x_values[-1]
    first_n, last_n = grid.y_values[0], grid.y_values[-1]
    assert axes[0][0].format_coord(np.max(grid.x_values)-np.min(grid.x_values), np.max(grid.y_values)-np.min(grid.y_values)).split("\n")[0]==f'bar={np.max(grid.x_values)}, n={(np.max(grid.y_values))}',"Wrong max"
    assert "COUNT = 497 " in fc(22, 0), "Wrong COUNT"
    assert axes[0][0].format_coord(0, 0).split("\n")[0]==f'bar={np.min(grid.x_values)}, n={(np.min(grid.y_values))}',"Wrong min "
    assert fc(x_edge - 0.01, 0).split("\n")[0] == f"bar={last_bar}, n={first_n}", "Last column, just inside"
    assert fc(x_edge, 0) == "Too far", "Last column, on the edge"
    assert fc(-0.5, 0).split("\n")[0] == f"bar={first_bar}, n={first_n}", "First column, on the edge"
    assert fc(-0.51, 0) == "Too far", "First column, just outside"
    assert fc(0, y_edge - 0.01).split("\n")[0] == f"bar={first_bar}, n={last_n}", "Last row, just inside"
    assert fc(0, y_edge) == "Too far", "Last row, on the edge"
    assert fc(0, -0.5).split("\n")[0] == f"bar={first_bar}, n={first_n}", "First row, on the edge"
    assert fc(0, -0.51) == "Too far", "First row, just outside"
    print(np.nanmin(grid.drawdown), np.nanmax(grid.drawdown))
    plt.show()
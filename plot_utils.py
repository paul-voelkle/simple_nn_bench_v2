#plotting setup
from matplotlib import pyplot as plt
import math
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import colorsys
import numpy as np
import os
from utilities import DataIO

class PlotSettings(DataIO):
    def __init__(self):
        super().__init__(filename="plot_config.pkl", path_default=".", name="Plotting Configuration", ver="0.0.1")   
        self.width=6
        self.height=5
        self.font = "Times New Roman"
        self.font_familiy = "serif"
        self.font_size_text = 16
        self.font_size_label = 16
        self.font_size_title = 22
        self.font_size_tick = 14
        self.scale_factor = 4
        self.figsize = (self.width*self.scale_factor, self.height*self.scale_factor)

        self.hist_density:bool = True
        self.hist_stacked:bool = True

        self.labelfont = FontProperties()
        self.axislabelfont = FontProperties()
        self.titlefont = FontProperties()
        self.tickfont = FontProperties()

        self.axisfontsize = self.font_size_text*self.scale_factor
        self.labelfontsize = self.font_size_text*self.scale_factor

        self.set_font()
        self.set_scale()

        self.__noedit__ = self.__noedit__ + ["labelfont", "axislabelfont", "titlefont", "tickfont", "axisfontsize", "labelfontsize"]

    
    def edit(self):
        super().edit()
        self.set_font()
        self.set_scale()      

    def set_font(self):

        self.labelfont.set_family(self.font_familiy)
        self.labelfont.set_name(self.font)
        self.labelfont.set_size(self.font_size_text*self.scale_factor)

        self.axislabelfont.set_family(self.font_familiy)
        self.axislabelfont.set_name(self.font)
        self.axislabelfont.set_size(self.font_size_label*self.scale_factor)

        self.titlefont.set_family(self.font_familiy)
        self.titlefont.set_name(self.font)
        self.titlefont.set_size(self.font_size_title*self.scale_factor)

        self.tickfont.set_family(self.font_familiy)
        self.tickfont.set_name(self.font)
        self.tickfont.set_size(self.font_size_tick*self.scale_factor)

        plt.rcParams["font.family"] = self.font_familiy
        plt.rcParams["mathtext.default"] = "rm"
        plt.rcParams['text.usetex'] = True

    def set_scale(self):
        # print(f"Setting scale to {self.scale_factor}")
        self.figsize = (self.width*self.scale_factor, self.height*self.scale_factor)
        self.labelfont.set_size(self.font_size_text*self.scale_factor)
        self.axislabelfont.set_size(self.font_size_label*self.scale_factor)  
        self.tickfont.set_size(self.font_size_tick*self.scale_factor)   
        self.titlefont.set_size(self.font_size_title*self.scale_factor)   
        self.axisfontsize = self.font_size_text*self.scale_factor
        self.labelfontsize = self.font_size_text*self.scale_factor
        plt.rcParams["font.size"] = self.font_size_tick*self.scale_factor


plot_settings = PlotSettings().load(PlotSettings().path_default)


def listify(x)->list:
    if not isinstance(x, list):
        return [x]  

def map_ticks(ticklabels, ticks_old):
    # print("Mapping ticks:")
    # print("Old tickmarks")
    # print(ticks_old)

    num_ticks = len(ticklabels)-1
    x_start = ticks_old[0] 
    x_stop = ticks_old[-1]
    x_step = abs(x_start-x_stop)/num_ticks

    ticks_new = np.arange(x_start, x_stop+x_step, x_step)
 
    return ticks_new



def create_plot(
            axs,
            fig,
            X_label:str="", 
            Y_label:str="", 
            X_scale="linear", 
            Y_scale="linear",
            xticks=[],
            yticks=[],         
            path='',
            fname='',
            title:str="",
            grid:bool=True
):
    #set axis labels, scale and tick format
    axs.set_xlabel( X_label, fontproperties=plot_settings.axislabelfont )
    axs.set_ylabel( Y_label, fontproperties=plot_settings.axislabelfont )

    axs.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
    
    axs.set_xscale( X_scale )
    axs.set_yscale( Y_scale )

    if len(xticks) != 0:
        xticks_old = [item for item in axs.get_xticks()]
        axs.set_xticks(map_ticks(xticks, xticks_old))
        axs.set_xticklabels(xticks)

    if len(yticks) != 0:
        yticks_old = [item for item in axs.get_yticks()]
        axs.set_yticks(map_ticks(yticks, yticks_old))
        axs.set_yticklabels(yticks)
        
    #set title
    axs.set_title(title, fontproperties=plot_settings.titlefont )
    
    #turn gird on
    if grid:
        axs.grid('on')
    
    #create legend
    axs.legend(loc='best', prop=plot_settings.labelfont, framealpha=0.0 )
    
    #save plot
    if fname != '':
        print(f"Saving plot to {path}/{fname}")
        os.makedirs(path, exist_ok=True)
        plt.savefig(fname=f"{path}/{fname}")

def plot_2d(
            x:list[np.ndarray], 
            y:list[np.ndarray]=[], 
            labels:list[str]="", 
            X_label:str="", 
            Y_label:str="", 
            X_scale:str="linear", 
            Y_scale:str="linear",
            xticks:list[float]=[],
            yticks:list[float]=[],         
            path:str='', 
            fname:str='',
            title:str="",
            linestyle:list[str]=["solid"],
            grid:bool=True,
            scale:float=4):
    
    plot_settings.scale_factor = scale
    plot_settings.set_scale()
    
    fig, axs = plt.figure(figsize=plot_settings.figsize), plt.axes()
    
    if not isinstance(x, list):
        x = [x]
        
    if not isinstance(labels, list):
        labels = [labels]
    
    if not isinstance(linestyle, list):
        linestyle = [linestyle]    
    
    if len(linestyle) != len(x):
        linestyle_old = linestyle
        linestyle = [linestyle_old[0]]
        for i in range(1,len(x)):
            linestyle.append(linestyle_old[i%len(linestyle_old)])
    
    for i in range(len(x)): 
        if len(x) == len(y):            
            axs.plot(x[i], y[i], label=labels[i], linestyle=linestyle[i])
        elif len(y)==1:
            axs.plot(x[i], y[0], label=labels[i], linestyle=linestyle[i])
        else:
            axs.plot(x[i], label=labels[i]) 
    

    create_plot(axs, fig, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, grid)

    return fig

def hist(
            x:list[np.ndarray], 
            labels:list[str]='', 
            X_label:str='', 
            Y_label:str='', 
            X_scale:str='linear', 
            Y_scale:str='linear',
            xticks:list[float]=[],
            yticks:list[float]=[],
            histtype:list[str]=['step'], 
            bins:int=0,
            path:str='',
            fname:str='', 
            title:str='',
            grid:bool=True,
            scale:float=4):

    plot_settings.scale_factor = scale
    plot_settings.set_scale()
    
    fig, axs = plt.figure(figsize=plot_settings.figsize), plt.axes()
    
    if not isinstance(x, list):
        x = [x]
    
    if not isinstance(labels, list):
        labels = [labels]    
    
    if not isinstance(histtype, list):
        histtype = [histtype]    
    
    if len(histtype) != len(x):
        histtype_old = histtype
        histtype = [histtype_old[0]]
        for i in range(1,len(x)):
            histtype.append(histtype_old[i%len(histtype_old)])
    
    if len(x) > 1:
        alpha = 1/len(x)
    else:
        alpha = 1.0
    
    for i in range(len(x)): 
        axs.hist(x[i], bins=bins, alpha=alpha, histtype=histtype[i], density=plot_settings.hist_density, stacked=plot_settings.hist_stacked, label=labels[i])   
        
    create_plot(axs, fig, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, grid)
    
    return fig
    

def scatter(
            x:list[np.ndarray], 
            y:list[np.ndarray], 
            labels:list[str]="", 
            X_label:str="", 
            Y_label:str="", 
            X_scale="linear", 
            Y_scale="linear", 
            xticks=[],
            yticks=[],        
            path='',
            fname='',
            title:str="",
            linestyle:list[str]=['solid'],
            grid:bool=True,
            scale:float=4):

    plot_settings.scale_factor = scale
    plot_settings.set_scale()
    
    fig, axs = plt.figure(figsize=plot_settings.figsize), plt.axes()
    
    if not isinstance(x, list):
        x = [x]

    if not isinstance(y, list):
        y = [y]
        
    if not isinstance(labels, list):
        labels = [labels]
        
    if not isinstance(linestyle, list):
        linestyle = [linestyle]
    
    if len(linestyle) != len(x):
        linestyle_old = linestyle
        linestyle = [linestyle_old[0]]
        for i in range(1,len(x)):
            linestyle.append(linestyle_old[i%len(linestyle_old)])
    
    for i in range(len(x)): 
        if len(x) == len(y):            
            axs.scatter(x[i], y[i], label=labels[i], linestyle=linestyle[i])
        elif len(y)==1:
            axs.scatter(x[i], y[0], label=labels[i], linestyle=linestyle[i])
        else:
            axs.scatter(x[i], label=labels[i]) 
    
    #axs.set_linestyle(linestyle)    
    
    create_plot(axs, fig, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, grid)

    return fig

def heatmap(
            x:list[np.ndarray], 
            labels:list[str]="", 
            X_label:str="", 
            Y_label:str="", 
            X_scale:str="linear", 
            Y_scale:str="linear", 
            xticks:list[float]=[],
            yticks:list[float]=[],        
            path:str='',
            fname:str='',
            title:str="",
            cmap:list[str]=['gist_heat_r'],
            grid:bool=True,
            scale:float = 4):

    plot_settings.scale_factor = scale
    plot_settings.set_scale()
    
    fig, axs = plt.figure(figsize=plot_settings.figsize), plt.axes()
    
    if not isinstance(x, list):
        x = [x]
    
    if not isinstance(labels, list):
        labels = [labels]    
    
    if not isinstance(cmap, list):
        cmap = [cmap]    
    
    if len(cmap) != len(x):
        cmap_old = cmap
        cmap = [cmap_old[0]]
        for i in range(1,len(x)):
            cmap.append(cmap_old[i%len(cmap_old)])
    
    for i in range(len(x)): 
        axs.imshow(x[i], label=labels[i], cmap=cmap[i]) 
    
    axs.set_ylim(ymin=40, ymax=0)
    axs.set_xlim(xmin=0, xmax=40)

    create_plot(axs, fig, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, grid)

    return fig

def multiplot(
            cols:int,
            rows:int, 
            plots:list,
            X_label:str="", 
            Y_label:str="", 
            X_scale="linear", 
            Y_scale="linear",
            xticks=[],
            yticks=[],         
            path='',
            fname='',
            title:str="",
            grid:bool=True):
    
    if cols*rows != len(plots):
        print("Error, columns*rows != len(plots)")
        return 
    
    if not isinstance(plots, list):
        plots = [plots]

    fig, axs = plt.subplots(rows, cols, figsize=(plot_settings.width*cols*plot_settings.scale_factor,plot_settings.height*rows*plot_settings.scale_factor))

    k = 0

    for i in range(rows):
        for j in range(cols):
            if rows != 1 and cols != 1: 
                axs[i,j] = plots[k]
            elif rows == 1:
                axs[j] = plots[k]
            elif cols == 1:
                axs[i] = plots[k]
            else:
                axs = plots[k]
            k += 1

    for ax in axs:
        ax.set_xlabel( X_label, fontproperties=plot_settings.axislabelfont )
        ax.set_ylabel( Y_label, fontproperties=plot_settings.axislabelfont )

        ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
        
        ax.set_xscale( X_scale )
        ax.set_yscale( Y_scale )

        if len(xticks) != 0:
            ax.set_xticklabels(xticks)
        if len(yticks) != 0:        
            ax.set_yticklabels(yticks)

        ax.legend(loc='best', prop=plot_settings.tickfont, framealpha=0.0 )

    if title != "" :
        fig.suptitle(title)

    if fname != "":
        print(f"Saving plot to {path}/{fname}")
        os.makedirs(path, exist_ok=True)
        plt.savefig(fname=f"{path}/{fname}")
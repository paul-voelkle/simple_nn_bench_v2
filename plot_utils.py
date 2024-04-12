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

width=6
height=5
font = "Times New Roman"
font_familiy = "serif"
font_size_text = 16
font_size_label = 18
font_size_title = 22
font_size_tick = 16
scale_factor = 4
figsize = (width*scale_factor, height*scale_factor)

labelfont = FontProperties()
labelfont.set_family(font_familiy)
labelfont.set_name(font)
labelfont.set_size(font_size_text*scale_factor)

axislabelfont = FontProperties()
axislabelfont.set_family(font_familiy)
axislabelfont.set_name(font)
axislabelfont.set_size(font_size_label*scale_factor)

titlefont = FontProperties()
titlefont.set_family(font_familiy)
titlefont.set_name(font)
titlefont.set_size(font_size_title*scale_factor)

tickfont = FontProperties()
tickfont.set_family(font_familiy)
tickfont.set_name(font)
tickfont.set_size(font_size_tick*scale_factor)

axisfontsize = font_size_text*scale_factor
labelfontsize = font_size_text*scale_factor

plt.rcParams["font.family"] = font_familiy
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams['text.usetex'] = True

def plot_helper(
        axs,
        fig,
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
        linestyle:list[str]=["solid"]
        ):
    #axs.set_linestyle(linestyle)

    axs.set_xlabel( X_label, fontproperties=axislabelfont )
    axs.set_ylabel( Y_label, fontproperties=axislabelfont )

    axs.set_title(title, fontproperties=titlefont)
    
    axs.set_xscale( X_scale )
    axs.set_yscale( Y_scale )


    #xticks = [ int(x) for x in axs.get_xticks() ]
    #axs.set_xticklabels( xticks, fontproperties=tickfont )

    #yticks = axs.get_yticks()
    #axs.set_yticklabels( yticks, fontproperties=tickfont )
    
    axs.legend(labels=labels, loc='best', prop=tickfont )
    
    if len(xticks) != 0:
        axs.set_xticklabels(xticks, fontproperties=tickfont)
    else:
        xticks = [ round(x_tick, 2) for x_tick in axs.get_xticks() ]
        axs.set_xticklabels(xticks, fontproperties=tickfont)
    
    if len(yticks) != 0:
        axs.set_yticklabels(yticks, fontproperties=tickfont)
    else:
        yticks = [ round(y_tick, 2) for y_tick in axs.get_yticks()]
        axs.set_yticklabels(yticks, fontproperties=tickfont)

    axs.legend(labels=labels, loc='best', prop=labelfont )
    
    #axs.tick_params(labelsize=labelfontsize)
    axs.grid('on')
    
    #fig.tight_layout()
        
    if fname != '':
        print(f"Saving plot to {path}/{fname}")
        os.makedirs(path, exist_ok=True)
        plt.savefig(fname=f"{path}/{fname}")
    
    return fig, axs    
    

def plot_2d(
        x:list[np.ndarray], 
        y:list[np.ndarray]=[], 
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
        linestyle:list[str]=["solid"]):
    
    fig, axs = plt.subplots(1,1, figsize=figsize)
    
    if not len(y)==0 and len(x) == len(y):
        for i in range(len(x)):
            axs.plot(x[i], y[i])
    else:
        for X in x:
            axs.plot(X) 
    
    plot_helper(axs, fig, labels, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, linestyle)
    
    return fig, axs

def hist(
        x:list[np.ndarray], 
        y:list[np.ndarray]=[], 
        labels:list[str]="", 
        X_label:str="", 
        Y_label:str="", 
        X_scale="linear", 
        Y_scale="linear",
        xticks=[],
        yticks=[],
        histtype="step", 
        bins:int=0,
        density:bool = False,
        path='',
        fname='', 
        title:str="",
        lables:list[str]=""
    ):
    fig, axs = plt.subplots(1, 1, figsize=figsize)

    axs.hist(x, bins=bins, histtype=histtype, density=density) 
    
    plot_helper(axs, fig, labels, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title)
    
    return fig, axs

def scatter(
        x:list[np.ndarray], 
        y:list[np.ndarray]=[], 
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
        linestyle:list[str]=''):
    
    fig, axs = plt.subplots(1,1, figsize=figsize)
    
    axs.scatter(x,y)
    
    plot_helper(axs, fig, labels, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, linestyle)
    
    return fig, axs

def heatmap(
        x:list[np.ndarray], 
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
        linestyle:list[str]=''
):
    fig, axs = plt.subplots(1,1, figsize=figsize)
    
    axs.imshow(x, cmap="gist_heat_r")

    plot_helper(axs, fig, labels, X_label, Y_label, X_scale, Y_scale, xticks, yticks, path, fname, title, linestyle)
    
    return fig, axs

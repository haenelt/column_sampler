"""
Make final plot

This script plots column height across layers for direct sequence comparison.

created by Daniel Haenelt
Date created: 01-07-2020             
Last modified: 11-10-2020  
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from fmri_tools.io.get_filename import get_filename

# input
file_ge = "/data/pt_01880/Experiment1_ODC/p3/analysis/line_plot/line_plot_GE_EPI23_Z_right_rest.npz"
file_se = "/data/pt_01880/Experiment1_ODC/p3/analysis/line_plot/line_plot_SE_EPI12_Z_right_rest.npz"
file_vaso = "/data/pt_01880/Experiment1_ODC/p3/analysis/line_plot/line_plot_VASO13_Z_right_rest.npz"
path_output = "/data/pt_01880/Experiment1_ODC/p3/analysis/final_plot"
name_output = None # invert data if set to vaso

# parameters
plot_rel = True

""" do not edit below """

# get contrast from input file
_, name_file, _ = get_filename(file_ge)
name_file = name_file.split("_")

unit = name_file[-3]
contrast = name_file[-2]+"_"+name_file[-1]

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# get number of layers
xc = int(len(np.load(file_ge)["x"])/2)
n_layer = len(np.load(file_ge)["y"])

# load data
ge_data = np.load(file_ge)
se_data = np.load(file_se)
vaso_data = np.load(file_vaso)

y_ge = ge_data["y"][:,xc]
y_se = se_data["y"][:,xc]
y_vaso = -1 * vaso_data["y"][:,xc]

yc_ge = ge_data["yc"][:,xc]
yc_se = se_data["yc"][:,xc]
yc_vaso = vaso_data["yc"][:,xc]

# relative to GM/WM border
if plot_rel:
    y_ge /= y_ge[0]
    y_se /= y_se[0]
    y_vaso /= y_vaso[0]
    
    yc_ge /= y_ge[0]
    yc_se /= y_se[0]
    yc_vaso /= y_vaso[0]
    
# specify colors
ge_color = (255/255,90/255,71/255)
se_color = (15/255,147/255,245/255)
vaso_color = (250/255,194/255,5/255)

"""
make height plot
"""
fig, ax = plt.subplots()
ax.set_facecolor("black") # set background color
ax.grid()

# plot
x = np.arange(n_layer) / (n_layer - 1)
ax.plot(x,y_ge, color=ge_color, label="GE-BOLD")
ax.plot(x,y_se, color=se_color, label="SE-BOLD")
ax.plot(x,y_vaso, color=vaso_color, label="VASO")

# confidence interval
ax.fill_between(x, 
                y_ge-yc_ge, 
                y_ge+yc_ge, 
                facecolor=(ge_color[0],ge_color[1],ge_color[2],.5), 
                edgecolor=(0,0,0,0),
                )

ax.fill_between(x, 
                y_se-yc_se, 
                y_se+yc_se, 
                facecolor=(se_color[0],se_color[1],se_color[2],.5), 
                edgecolor=(0,0,0,0),
                )


ax.fill_between(x, 
                y_vaso-yc_vaso, 
                y_vaso+yc_vaso, 
                facecolor=(vaso_color[0],vaso_color[1],vaso_color[2],.5), 
                edgecolor=(0,0,0,0),
                )

# set y-label
ylabel = r'$z\,/\,z_{wm}$ ' if plot_rel == True else r'$z$-score '
ylabel += "("+name_file[-2]+" > "+name_file[-1]+")"

# labels
ax.set(xlabel=r'$WM\,\to\,CSF$',
       ylabel=ylabel,
       )
ax.legend(loc=2)
   
# save plot
suffix = "_"+name_output if name_output else ""
suffix += "_rel" if plot_rel == True else "_abs"
fig.savefig(os.path.join(path_output,"column_height_final_"+unit+"_"+contrast+suffix+".png"), dpi=250)
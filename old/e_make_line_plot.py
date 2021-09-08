"""
This script plots the averaged sampled data at different cortical depths.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import sem
from fmri_tools.io.get_filename import get_filename
from lib_column.plot.get_width import get_width
from lib_column.plot.get_confidence_interval import get_confidence_interval

# input
file_data = ["/data/pt_01880/Experiment1_ODC/p1/analysis/data_sampling/lh_data_sampling_GE_EPI3_Z_left_right.npz",
             "/data/pt_01880/Experiment1_ODC/p1/analysis/data_sampling/rh_data_sampling_GE_EPI3_Z_left_right.npz",
             "/data/pt_01880/Experiment1_ODC/p1/analysis/data_sampling/lh_data_sampling_GE_EPI4_Z_left_right.npz",
             "/data/pt_01880/Experiment1_ODC/p1/analysis/data_sampling/rh_data_sampling_GE_EPI4_Z_left_right.npz",
             ] # list of npz with sampled data
path_output = "/data/pt_01880/Experiment1_ODC/p1/analysis/line_plot3"
name_output = None

# parameters
plot_width = True
line_length = 2 # in mm
line_step = 0.1 # in mm
confidence = 0.95
ylim = [-0.4,1.5]

""" do not edit below """

# get contrast and unit
_, name_file, _ = get_filename(file_data[0])
name_file = name_file.split("_")
unit = name_file[-3]
contrast = name_file[-2]+"_"+name_file[-1]

# get session name
number_sess = []
number_sess2 = []
for i in range(len(file_data)):
    _, name_file, _ = get_filename(file_data[i])
    name_file = name_file.split("_")
    number_sess.append(name_file[-4][-1])
number_sess = list(set(number_sess))
number_sess = list(map(int, number_sess))
number_sess.sort()
number_sess2 = [str(x) for x in number_sess]
number_sess2 = "".join(number_sess2)

if name_file[-4][0] == "V":
    name_sess = name_file[-4][:-1]+number_sess2
    invert_peak = True
else:
    invert_peak = False
    name_sess = name_file[-5]+"_"+name_file[-4][:-1]+number_sess2

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

path_img = os.path.join(path_output,"img",contrast)
if not os.path.exists(path_img):
    os.makedirs(path_img)

# load line data
contrast_data = np.load(file_data[0], allow_pickle=True)
x = np.arange(-line_length,line_length+line_step,line_step)
y = contrast_data["arr_val"]

# get array size
n_layer = np.shape(y)[1]
n_coords = np.shape(y)[2]

# print infos
print("# of line coordinates: "+str(n_coords))
print("# of layers: "+str(n_layer))

# initialize arrays
n_elements = 0
for i in range(len(file_data)):
    n_elements += len(np.load(file_data[i], allow_pickle=True)["arr_val"])

y = np.zeros((n_elements, n_layer, n_coords))
y_mean = np.zeros((n_layer, n_coords))
y_confidence = np.zeros((n_layer, n_coords))

# get mean and confidence interval
for i in range(n_layer):
    for j in range(n_coords):
        for k in range(len(file_data)):
            contrast_data = np.load(file_data[k], allow_pickle=True)["arr_val"]
            contrast_length = len(contrast_data)

            counter = 0            
            for q in range(contrast_length):
                y[counter,:,:] = contrast_data[counter,:,:]
                counter += 1
                
        y_mean[i,j] = np.nanmean(y[:,i,j])      
        y_confidence[i,j] = get_confidence_interval(sem(y[:,i,j], nan_policy="omit"), 
                                                    len(y[:,i,j][~np.isnan(y[:,i,j])]), 
                                                    confidence)

# get left and right column width
if plot_width:
    ll, lr = get_width(x, y_mean, invert_peak)

# plot only fraction of layers
frac = 2

# make plot
fig = plt.figure(figsize=(4, 3))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
ax_bar = fig.add_axes([0.97, 0.05, 0.05, 0.9])

ax.set_prop_cycle(color=[plt.cm.summer(i) for i in np.linspace(0,1,int(n_layer/frac))])
ax.set_facecolor("black")
for i in range(0,n_layer,frac):
    ax.plot(x, y_mean[i,:], label="layer "+str(i))
    
    # plot confidence interval
    ax.fill_between(x, 
                    y_mean[i,:]-y_confidence[i,:], 
                    y_mean[i,:]+y_confidence[i,:],
                    alpha=0.3)
    
    # plot column width
    if plot_width:
        ax.axvline(ll,0,1, color="white", linestyle="--")
        ax.axvline(lr,0,1, color="white", linestyle="--")

# set y-label
unit_label = r"$z$-score " if unit == "Z" else r"percent signal change in p.u. "

if contrast == "left_right":
    contrast_label = "(left > right)"
elif contrast == "left_rest":
    contrast_label = "(left > rest)"
else:
    contrast_label = "(right > rest)"

# y-axis
if ylim:
    ax.set_ylim(ylim)

# add labels
ax.set_xlabel("displacement in mm", fontsize=14)
ax.set_ylabel(unit_label+contrast_label, fontsize=14)
ax.grid()

# add ticks
ax.tick_params(axis='both', which='major', labelsize=12)

# add colorbar
c_bar = mpl.colorbar.ColorbarBase(ax_bar, 
                                  cmap=mpl.cm.summer,
                                  orientation='vertical')
c_bar.set_ticks([])
c_bar.set_label(r'$WM\,\to\,CSF$', fontsize=14)

# save plot
suffix = "_"+name_output if name_output else ""
fig.savefig(os.path.join(path_img,"line_plot_"+name_sess+"_"+unit+"_"+contrast+suffix+".png"), 
            format="png", 
            bbox_inches='tight', 
            dpi=250)

# save data
np.savez(os.path.join(path_output,"line_plot_"+name_sess+"_"+unit+"_"+contrast+suffix), 
         x=x,
         y=y_mean,
         yc=y_confidence,
         )
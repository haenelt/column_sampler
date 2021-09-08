"""
This script takes the column array and samples contrast data for each coordinate.
"""
import os
import numpy as np
from nibabel.affines import apply_affine
from gbb.interpolation.linear_interpolation3d import linear_interpolation3d
from lib_column.io.LoadData import LoadData

# input
participant = "p3"
hemi = "rh"
name_sess = "VASO3"
unit = "Z" # Z or raw
contrast = "right_rest" # left_right, left_rest or right_rest
column_in = "/data/pt_01880/Experiment1_ODC/p3/analysis/column_array/rh_column_array.npz"
path_output = "/data/pt_01880/Experiment1_ODC/p3/analysis/data_sampling"
name_output = None

""" do not edit below """

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# load
data = LoadData(participant, None, name_sess, unit, contrast)
column = np.load(column_in, allow_pickle=True)
cmap = data.get_deformation()["data"]
source = data.get_native()["data"]

arr_column = column["arr_column"]
arr_cmap = cmap.get_fdata()
arr_source = source.get_fdata()

# ras to voxel transformation matrix
ras2vox_tkr = data.get_matrix()["ras2vox"]

# initialize
arr_val = np.zeros_like(arr_column[:,:,:,0])

n_elements = np.shape(arr_column)[0]
n_layer = np.shape(arr_column)[1]
n_coords = np.shape(arr_column)[2]

# print infos
print("# of line elements: "+str(n_elements))
print("# of line coordinates: "+str(n_coords))
print("# of layers: "+str(n_layer))

# get source image dimensions
x_dim = np.shape(arr_source)[0]
y_dim = np.shape(arr_source)[1]
z_dim = np.shape(arr_source)[2]

# sample data
for i in range(n_elements):
    for j in range(n_layer):
        for k in range(n_coords):
            
            # get coordinates in voxel space
            temp = arr_column[i,j,k,:]
            temp = apply_affine(ras2vox_tkr, temp)                    
        
            # transform coordinates to source space
            x = linear_interpolation3d(temp[0], temp[1], temp[2], arr_cmap[:,:,:,0])
            y = linear_interpolation3d(temp[0], temp[1], temp[2], arr_cmap[:,:,:,1])
            z = linear_interpolation3d(temp[0], temp[1], temp[2], arr_cmap[:,:,:,2])

            if x < 0 or x > x_dim - 1:
                arr_val[i,j,k] = np.nan
                continue
            
            if y < 0 or y > y_dim - 1:
                arr_val[i,j,k] = np.nan
                continue
            
            if z < 0 or z > z_dim - 1:
                arr_val[i,j,k] = np.nan
                continue
            
            # sample source data
            arr_val[i,j,k] = linear_interpolation3d(x, y, z, arr_source)
            
# save sampled data
suffix = "_"+name_output if name_output else ""
np.savez(os.path.join(path_output, hemi+"_data_sampling_"+name_sess+"_"+unit+"_"+contrast+suffix), arr_val=arr_val)

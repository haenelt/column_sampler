"""
This script sorts all lines and computes corresponding cortical depth dependent coordinates. The
array has the dimensions line x coordinates x layer.
"""
import os
import numpy as np
import nibabel as nb
from nibabel.freesurfer.io import read_geometry
from gbb.normal.get_normal import get_normal
from lib_column.io.LoadData import LoadData

participant = "p3"
line_in = "/data/pt_01880/Experiment1_ODC/p3/analysis/line_coordinates/rh_line_coordinates.npz"
csf_mesh_in = None
wm_mesh_in = None
path_output = "/data/pt_01880/Experiment1_ODC/p3/analysis/column_array"
name_output = None

# parameters
hemi = "rh"
n_layer = 20

""" do not edit below """

def min_distance(coords, pts):
    coords = coords - pts
    r = coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2
    r = np.sqrt(r)
    ind = np.where(r == np.min(r))[0][0]
    
    return ind

def get_depth(y_start, y_end, n_layer, layer):
    return ( y_end - y_start ) * layer/(n_layer-1) + y_start

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# load
data = LoadData(participant, hemi, None, None, None)
data_line = np.load(line_in, allow_pickle=True)

line = data_line["line"]
mesh = data.get_mesh()
rim = data.get_rim()
normal = get_normal(mesh["vtx"], mesh["fac"])

# get border: do not use this function: get other layers from meshlines
if csf_mesh_in:
    vtx_csf, _ = read_geometry(csf_mesh_in)
else:
    vtx_csf, _ = get_border(mesh["name"], 
                            rim["name"], 
                            line_length=3, 
                            n_neighbor=3, 
                            direction="csf", 
                            path_output=path_output, 
                            write_output=True)

if wm_mesh_in:
    vtx_wm, _ = read_geometry(wm_mesh_in)
else:
    vtx_wm, _ = get_border(mesh["name"], 
                           rim["name"], 
                           line_length=3, 
                           n_neighbor=3, 
                           direction="wm", 
                           path_output=path_output, 
                           write_output=True)

# print infos
n_line = len(line)
n_elements = np.sum([len(line[i]) for i in range(len(line))])
n_coords = len(line[0][0])

print("# of lines: "+str(n_line))
print("# of line elements: "+str(n_elements))
print("# of line coordinates: "+str(n_coords))
print("# of layers: "+str(n_layer))

# set header
affine = np.eye(4)
header = nb.Nifti1Header()

# initialize arrays
arr_column = np.zeros((n_elements, n_layer, n_coords, 3))
arr_data = np.zeros((len(mesh["vtx"]), 1, 1))

element = 0
for i in range(n_line):
    for j in range(len(line[i])):
        for p in range(n_coords):
            
            # get index of nearest vtx
            ind = min_distance(mesh["vtx"], line[i][j][p])
            
            # mark in overlay for sanity check 
            arr_data[ind] = 1
            
            for q in range(n_layer):
                
                # write coordinates into array
                arr_column[element,q,p,:] = get_depth(vtx_wm[ind,:], vtx_csf[ind,:], n_layer, q)
    
        element += 1
            
# save column array
suffix = "_"+name_output if name_output else ""
np.savez(os.path.join(path_output, hemi+"_column_array"+suffix), arr_column=arr_column)

# write overlay to see if right neighbors are selected
output = nb.Nifti1Image(arr_data, affine, header)
nb.save(output, os.path.join(path_output, hemi+".column_array"+suffix+".mgh"))

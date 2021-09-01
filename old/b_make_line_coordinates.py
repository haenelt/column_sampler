"""
Make line coordinates

This script computes the lines perpendicular to the manually defined path along a cortical column.
Reference data is used to shift every single line to have the column peak at its center coordinate.
Line coordinates and geometry files are saved.

created by Daniel Haenelt
Date created: 06-03-2020             
Last modified: 05-10-2020  
"""
import os
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from gbb.utils.get_adjm import get_adjm
from lib_column.io.LoadData import LoadData
from lib_column.io.write_line import write_line
from lib_column.utils.get_perpendicular_line import get_perpendicular_line
from lib_column.utils.sample_line import sample_line
from lib_column.utils.get_line_shift import get_line_shift
from lib_column.utils.apply_line_shift import apply_line_shift

# input
participant = "p3"
file_path = "/data/pt_01880/Experiment1_ODC/p3/analysis/path/rh.path"
file_ref = "/data/pt_01880/Experiment1_ODC/p3/analysis/template/vol_average_Z_left_right.nii"
path_output = "/data/pt_01880/Experiment1_ODC/p3/analysis/line_coordinates"
name_output = None

# parameters
hemi = "rh"
line_length = 2
line_step = 0.1
nn_smooth = 0
ndir_smooth = 5

""" do not edit below """

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# load mesh dictionary
mesh = LoadData(participant, hemi, None, None, None).get_mesh()

# x-coordinates
x = np.arange(-line_length,line_length+line_step,line_step)

# load reference volume
ref_array = nb.load(file_ref).get_fdata()

# get adjavency matrix
adjm = get_adjm(mesh["vtx"], mesh["fac"])

# get perpendicular lines
line = get_perpendicular_line(file_path, mesh["vtx"], mesh["fac"], adjm, line_length, line_step, 
                              nn_smooth, ndir_smooth)

# get and apply line shift
for i in range(len(line)):    
    for j in range(len(line[i])):

        # sample reference data
        data = sample_line(line[i][j], file_ref, ref_array)
        
        # get shift
        shift = get_line_shift(data)
        print("Shift for point "+str(j)+": "+str(shift))
        
        # apply shift to line
        line[i][j] = apply_line_shift(line[i][j], shift, line_length, line_step)
        
        # sample data to shifted line for sanity check
        data = sample_line(line[i][j], file_ref, ref_array)
        plt.plot(x, data)

# save corrected lines
suffix = "_"+name_output if name_output else ""
np.savez(os.path.join(path_output, hemi+"_line_coordinates"+suffix), x=x, line=line)
        
# save line geometry
for i in range(len(line)):

    # make output folder
    path_line = os.path.join(path_output,"line",hemi,str(i))    
    if not os.path.exists(path_line):
        os.makedirs(path_line)
    
    for j in range(len(line[i])):
        write_line(line[i][j], path_line, f"{j:04}")

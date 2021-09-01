"""
Make line deformation

This script deforms the line coordinates to a data set of another session using a coordinate
mapping.

created by Daniel Haenelt
Date created: 10-03-2020             
Last modified: 19-10-2020  
"""
import os
import numpy as np
from nibabel.freesurfer.io import read_geometry
from fmri_tools.io.get_filename import get_filename
from fmri_tools.surface.deform_surface import deform_surface

path_data = "/data/pt_01880/odc_analysis_test"
source_in = "/data/pt_01880/temp_odc/deformation/ge_epi3/temp/epi_source/epi.nii"
target_in = "/data/pt_01880/temp_odc/deformation/ge_epi3/temp/epi_target/epi.nii"
deform_in = "/data/pt_01880/temp_odc/deformation/ge_epi3/target2source.nii.gz" # target2source
name_sess_in = "GE_EPI2"
name_sess_out = "GE_EPI3"

# parameters
nlabel = 2
hemi = "lh"
line_length = 2
line_step = 0.1

""" do not edit below """

# x-coordinates
x = np.arange(-line_length,line_length+line_step,line_step)

line = []
for i in range(nlabel):
    
    # get input path
    path_input = os.path.join(path_data, "line", name_sess_in, str(i))
    
    # make output folder
    path_output = os.path.join(path_data, "line", name_sess_out, str(i))  
    if not os.path.exists(path_output):
        os.makedirs(path_output)    

    # load input surfaces
    surf_in = sorted(os.listdir(path_input))
    
    # deform surface
    line_temp = []
    for j in range(len(surf_in)):
        
        # get filename
        file_in = os.path.join(path_input, hemi+"."+surf_in[j])
        _, surf_out, _ = get_filename(file_in)
        
        deform_surface(input_surf = file_in, 
                       input_orig = source_in, 
                       input_deform = deform_in,
                       input_target = target_in,
                       path_output = path_output,
                       interp_method = "trilinear",
                       smooth_iter = 0,
                       sort_faces = False, 
                       flip_faces = False,
                       cleanup = True)
        
        # rename output
        file_out = os.path.join(path_output, surf_out)
        os.rename(file_out+"_def", file_out)

        # append line coordinates
        vtx, _ = read_geometry(file_out)
        line_temp.append(vtx)
    
    # append label lines
    line.append(line_temp)

# save deformed lines
np.savez(os.path.join(path_data, "line_coordinates_"+name_sess_out), x=x, line=line)
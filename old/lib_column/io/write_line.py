def write_line(vtx, path_output, name_output):
    """
    This function writes a surface file for an array of line coordinates.
    Inputs:
        *vtx: array of vertex coordinates.
        *path_output: path where output is written.
        *name_output: basename of output file.
        
    created by Daniel Haenelt
    Date created: 05-03-2020  
    Last modified: 05-03-2020
    """
    import os
    import numpy as np
    from nibabel.freesurfer.io import write_geometry

    # make output folder
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    
    # define faces
    fac = np.zeros((len(vtx)-1,3))
    for i in range(len(vtx)-1):
        fac[i,:] = [i,i+1,i]
    
    # write output
    write_geometry(os.path.join(path_output, name_output), vtx, fac)
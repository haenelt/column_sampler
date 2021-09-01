def get_border(mesh_in, rim_in, line_length=3, n_neighbor=3, direction="csf", path_output=None, 
               write_output=False):
    """
    Based on a rim file, new vertices are shifted from a mesh corresponding to the mid-cortical 
    surface in normal direction either to the GM/WM or GM/CSF border. A new file is written as 
    freesurfer mesh.
    Inputs:
        *mesh_in: filename of input mesh.
        *rim_in: filename of rim file (as used in laynii).
        *line_length: length of line in normal direction (in mm).
        *n_neighbor: number of neighbor for smoothing in normal direction.
        *direction: choose border (csf or wm).
        *path_output: path where output is written.
        *write_output: write final mesh to file (optional).
    Outputs:
        *vtx_final: new vertex points.
        *fac: corresponding faces.

    created by Daniel Haenelt
    Date created: 24-06-2020             
    Last modified: 13-10-2020
    """
    import os
    import numpy as np
    import nibabel as nb
    from nibabel.affines import apply_affine
    from nibabel.freesurfer.io import read_geometry, write_geometry
    from gbb.utils.vox2ras import vox2ras
    from gbb.normal.get_normal import get_normal
    from gbb.interpolation.nn_interpolation3d import nn_interpolation3d
    from gbb.utils.get_adjm import get_adjm
    from gbb.neighbor.nn_2d import nn_2d

    # fixed parameters
    n_line = 1000 # number of line points

    # load data
    vtx, fac = read_geometry(mesh_in)
    normal = get_normal(vtx, fac)

    rim = nb.load(rim_in)
    arr_rim = rim.get_fdata()

    # get adjacency matrix
    adjm = get_adjm(vtx, fac)

    # get header transformation
    vox2ras_tkr, ras2vox_tkr = vox2ras(rim_in)

    # get hemisphere from mesh basename
    hemi = os.path.splitext(os.path.basename(mesh_in))[0]

    # get volume dimensions
    xdim = np.shape(arr_rim)[0]
    ydim = np.shape(arr_rim)[1]
    zdim = np.shape(arr_rim)[2]

    if direction == "csf":
        vtx_end = vtx + line_length * normal
    elif direction == "wm":
        vtx_end = vtx - line_length * normal

    print("get normal lines")
    line_weights = np.linspace((0,0,0),(1,1,1,),n_line,dtype=np.float)
    vtx_all = [(vtx_end-vtx) * line_weights[i] + vtx for i in range(n_line)]
    vtx_all = [apply_affine(ras2vox_tkr, vtx_all[i]) for i in range(n_line)]
    vtx_all = np.array(vtx_all)

    print("remove outliers")
    vtx_all[:,:,0][vtx_all[:,:,0] < 0] = 0
    vtx_all[:,:,1][vtx_all[:,:,1] < 0] = 0
    vtx_all[:,:,2][vtx_all[:,:,2] < 0] = 0

    vtx_all[:,:,0][vtx_all[:,:,0] > xdim - 1] = xdim - 1
    vtx_all[:,:,1][vtx_all[:,:,1] > ydim - 1] = ydim - 1
    vtx_all[:,:,2][vtx_all[:,:,2] > zdim - 1] = zdim - 1

    print("sample volume data")
    rim_sampled = [nn_interpolation3d(vtx_all[i,:,0], vtx_all[i,:,1], vtx_all[i,:,2], arr_rim) for i in range(n_line)]
    rim_sampled = np.array(rim_sampled)

    print("find border")
    rim_border = np.zeros(len(vtx))
    for i in range(len(vtx)):
    
        counter = 0
        while True:
        
            counter += 1        
            if rim_sampled[counter, i] != 3:
                rim_border[i] = counter
                break
            elif counter == n_line - 1:
                rim_border[i] = 0
                break

    print("finalize")
    rim_border_final = [np.mean(rim_border[nn_2d(i, adjm, n_neighbor)]) for i in range(len(vtx))]
    rim_border_final = np.round(rim_border_final).astype(int).tolist()
    vtx_final = np.array([vtx_all[rim_border_final[i],i,:] for i in range(len(vtx))])
    vtx_final = apply_affine(vox2ras_tkr, vtx_final)

    # write output
    if write_output:
        write_geometry(os.path.join(path_output,hemi+"."+direction), vtx_final, fac)
        
    return vtx_final, fac
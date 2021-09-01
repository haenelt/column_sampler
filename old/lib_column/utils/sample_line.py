def sample_line(coords, file_data, data_array):
    """
    This function samples data onto the given coordinate array using linear interpolation.
    Inputs:
        *coords: coordinate array.
        *file_data: filename of nifti volume.
        *data_array: array with data points of nifti volume.
    Outputs:
        *data_sampled: array with sampled data.
        
    created by Daniel Haenelt
    Date created: 05-03-2020         
    Last modified: 13-10-2020
    """
    from nibabel.affines import apply_affine
    from gbb.interpolation.linear_interpolation3d import linear_interpolation3d
    from gbb.utils.vox2ras import vox2ras
    
    # get ras to voxel transformation
    _, ras2vox_tkr = vox2ras(file_data)

    # apply transformation to input coordinates
    coords = apply_affine(ras2vox_tkr, coords)

    # sample data
    data_sampled = linear_interpolation3d(coords[:,0],
                                          coords[:,1],
                                          coords[:,2], 
                                          data_array)
    
    return data_sampled
import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine


def make_template(arr, threshold=1.7):
    """
    *input list of arrays
    *for each array nan values below threshold
    *for all arrays nan values which have inconsistent signs
    """
    nrows, ncols = np.shape(arr)
    res = np.ones(nrows)

    if threshold:
        for i in range(ncols):
            res[arr[:, i] < threshold] = 0

    for i in range(ncols):
        tmp = np.sign(arr[:, 0]) * np.sign(arr[:, i])
        res[tmp != 1] = 0

    return res


def linear_interpolation3d(x, y, z, arr_c):
    """Apply a linear interpolation of values in a 3D volume to an array of
    coordinates.
    Parameters
    ----------
    x : (N,) np.ndarray
        x-coordinates in voxel space.
    y : (N,) np.ndarray
        y-coordinates in voxel space.
    z : (N,) np.ndarray
        z-coordinates in voxel space.
    arr_c : (U,V,W) np.ndarray
        3D array with input values.
    Returns
    -------
    c : (N,) np.ndarray
        Interpolated values for [x,y,z].
    """

    # corner points
    x0 = np.floor(x).astype(int)
    x1 = np.ceil(x).astype(int)
    y0 = np.floor(y).astype(int)
    y1 = np.ceil(y).astype(int)
    z0 = np.floor(z).astype(int)
    z1 = np.ceil(z).astype(int)

    # distances to corner points
    xd = [_careful_divide(x[i], x0[i], x1[i]) for i, _ in enumerate(x)]
    yd = [_careful_divide(y[i], y0[i], y1[i]) for i, _ in enumerate(y)]
    zd = [_careful_divide(z[i], z0[i], z1[i]) for i, _ in enumerate(z)]

    xd = np.asarray(xd)
    yd = np.asarray(yd)
    zd = np.asarray(zd)

    # corner values
    c000 = arr_c[x0, y0, z0]
    c001 = arr_c[x0, y0, z1]
    c010 = arr_c[x0, y1, z0]
    c011 = arr_c[x0, y1, z1]
    c100 = arr_c[x1, y0, z0]
    c101 = arr_c[x1, y0, z1]
    c110 = arr_c[x1, y1, z0]
    c111 = arr_c[x1, y1, z1]

    # interpolation along x-axis
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    # interpolation along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # interpolation along z-axis
    c = c0 * (1 - zd) + c1 * zd

    return c


def _careful_divide(v, v0, v1):
    """Only divide if v0 and v1 are different from each other."""

    return (v - v0) / (v1 - v0) if v1 != v0 else v


def ras2vox(vtx, vol_in):
    """
    https://neurostars.org/t/get-voxel-to-ras-transformation-from-nifti-file/4549
    """

    nii = nb.load(vol_in)
    mgh = nb.MGHImage(nii.dataobj, nii.affine)

    vox2ras_tkr = mgh.header.get_vox2ras_tkr()
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)

    return apply_affine(ras2vox_tkr, vtx)  # vox2ras_tkr


def sample_line(vtx, vol_in, deform_in):
    """
    This function samples data onto the given coordinate array using linear interpolation.
    Inputs:
        *vtx: vertex coordinates in voxel space.
        *file_data: filename of nifti volume.
        *data_array: array with data points of nifti volume.
    Outputs:
        *res: array with sampled data.

    """

    vtx = ras2vox(vtx, vol_in)
    arr = nb.load(vol_in).get_fdata()
    vtx_new = np.zeros_like(vtx)
    if deform_in is not None:
        arr_cmap = nb.load(deform_in).get_fdata()
        for i in range(3):
            vtx_new[:,i] = linear_interpolation3d(vtx[:,0], vtx[:,1], vtx[:,2],
                                                  arr_cmap[:,:,:,i])
        vtx = vtx_new.copy()

    # sample data
    res = linear_interpolation3d(vtx[:, 0], vtx[:, 1], vtx[:, 2], arr)

    return res

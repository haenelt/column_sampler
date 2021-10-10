# -*- coding: utf-8 -*-
"""Utility functions."""

import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine

__all__ = ["make_template", "sample_data", "flatten_array", "unflatten_array"]


def make_template(arr, threshold=1.7):
    """Get a mask from a set of contrast arrays. A two-dimensional array
    containing individual contrast arrays in separate column is used as input. A
    mask is created by excluding all data points which are below threshold in at
    least one contrast array (column). Finally, all data points are excluded
    which have inconsistent signs between contrasts (columns).

    Parameters
    ----------
    arr : np.ndarray, shape=(N, M)
        Contrast array.
    threshold : float, optional
        Threshold value.

    Returns
    -------
    mask : np.ndarray, shape=(N,)
        Mask array.

    """

    nrows, ncols = np.shape(arr)
    mask = np.ones(nrows)

    if threshold:
        for i in range(ncols):
            mask[arr[:, i] < threshold] = 0

    for i in range(ncols):
        tmp = np.sign(arr[:, 0]) * np.sign(arr[:, i])
        mask[tmp != 1] = 0

    return mask


def sample_data(vtx, vol_in, deform_in=None):
    """Samples volume data for a given array of vertex points using linear
    interpolation. Vertices are expected to be in tksurfer RAS space.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N, 3)
        Vertex array in tksurfer RAS space.
    vol_in : str
        Reference image (nifti volume).
    deform_in : str, optional
        Deformation field (4D nifti volume).

    Returns
    -------
    np.ndarray, shape=(N,)
        Sampled data.

    """

    arr = nb.load(vol_in).get_fdata()

    # transform to voxel space
    vtx = _ras2vox(vtx, vol_in)

    # apply deformation
    vtx_trans = np.zeros_like(vtx)
    if deform_in:
        arr_deform = nb.load(deform_in).get_fdata()
        for i in range(3):
            vtx_trans[:, i] = _linear_interpolation3d(vtx[:, 0],
                                                      vtx[:, 1],
                                                      vtx[:, 2],
                                                      arr_deform[:, :, :, i])
        vtx = vtx_trans.copy()

    return _linear_interpolation3d(vtx[:, 0], vtx[:, 1], vtx[:, 2], arr)


def flatten_array(pts):
    """Flattens a coordinate array of shape (N, M, 3) => (N*M, 3) or data array
    of shape (N, M) => (N*M,).

    Parameters
    ----------
    pts : np.ndarray, shape=(N, M, 3) or shape=(N, M)
        Coordinate array or data array.

    Returns
    -------
    np.ndarray, shape=(N*M, 3) or shape=(N*M,)
        Flattened coordinate array or data array.

    """

    pts = np.array(pts)
    dim1, dim2 = np.shape(pts)[:2]
    if np.shape(pts)[2] == 3 and len(np.shape(pts)) == 3:
        return np.reshape(pts, (dim1 * dim2, 3))
    else:
        return np.reshape(pts, (dim1 * dim2))


def unflatten_array(pts, pts_ref):
    """Unflattens a coordinate array of shape (N*M, 3) => (N, M, 3) or data
    array of shape (N*M,) => (N, M).

    Parameters
    ----------
    pts : np.ndarray, shape=(N*M, 3) or shape=(N*M,)
        Flattened coordinate array or data array.
    pts_ref : np.ndarray, shape=(N, M, ...)
        Reference array.

    Returns
    -------
    np.ndarray, shape=(N, M, 3) or shape=(N, M)
        Unflattened coordinate array or data array.

    """

    pts = np.array(pts)
    pts_ref = np.array(pts_ref)
    dim1, dim2 = np.shape(pts_ref)[:2]
    if np.shape(pts)[-1] == 3 and len(np.shape(pts)) == 2:
        return np.reshape(pts, (dim1, dim2, 3))
    else:
        return np.reshape(pts, (dim1, dim2))


def _linear_interpolation3d(x, y, z, vol):
    """Apply a linear interpolation of values in a 3D array to an array of
    coordinates.

    Parameters
    ----------
    x : np.ndarray, shape=(N,)
        x-coordinates in voxel space.
    y : np.ndarray, shape=(N,)
        y-coordinates in voxel space.
    z : np.ndarray, shape=(N,)
        z-coordinates in voxel space.
    vol : np.ndarray, shape=(U, V, W)
        3D array with input values.

    Returns
    -------
    c : np.ndarray, shape=(N,)
        Interpolated values for (x, y, z) coordinates.

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
    c000 = vol[x0, y0, z0]
    c001 = vol[x0, y0, z1]
    c010 = vol[x0, y1, z0]
    c011 = vol[x0, y1, z1]
    c100 = vol[x1, y0, z0]
    c101 = vol[x1, y0, z1]
    c110 = vol[x1, y1, z0]
    c111 = vol[x1, y1, z1]

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


def _careful_divide(v1, v2, v3):
    """Only divide if v2 and v3 are different from each other.

    Parameters
    ----------
    v1 : float
        Coordinate 1.
    v2 : float
        Coordinate 2.
    v3 : float
        Coordinate 3.

    Returns
    -------
    float
        Division result.

    """

    return (v1 - v2) / (v3 - v2) if v3 != v2 else v1


def _ras2vox(vtx, vol_in):
    """Transforms a vertex array from tksurfer RAS coordinates to voxel
    coordinates based on a reference nifti volume. The code is based on [1]_.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N, 3)
        Vertex array in tksurfer space.
    vol_in : str
        Reference nifti volume.

    Returns
    -------
    np.ndarray, shape=(N, 3)
        Vertex array in voxel space.

    References
    ----------
    .. [1] https://neurostars.org/t/get-voxel-to-ras-transformation-from-nifti-file/4549

    """

    nii = nb.load(vol_in)
    mgh = nb.MGHImage(nii.dataobj, nii.affine)

    vox2ras_tkr = mgh.header.get_vox2ras_tkr()
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)

    return apply_affine(ras2vox_tkr, vtx)

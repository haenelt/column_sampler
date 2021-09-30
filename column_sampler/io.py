# -*- coding: utf-8 -*-
"""I/O functions."""

import os
import numpy as np
from nibabel.freesurfer.io import write_geometry
from column_filter.io import save_overlay
from util import flatten_array

__all__ = ["read_anchor", "load_coords", "load_data", "save_coords",
           "save_data", "save_meshlines", "coords_to_mesh", "data_to_overlay"]


def read_anchor(file_in):
    """Load a text file containing a list of vertex indices. Individual indices
    must be separated by commas. Each row of indices is returned in a separate
    list.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    FileNotFoundError
        If `file_in` is not an existing file name.

    Returns
    -------
    res : dict
        Dictionary containing a list of indices for each row found in the text
        file. Row numbers are used as keys.
    """

    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    with open(file_in) as fp:
        rows = fp.readlines()
        res = {}
        for i, r in enumerate(rows):
            tmp = r.strip()
            res[str(i)] = [int(e) if e.isdigit() else e for e in tmp.split(',')]

    return res


def load_coords(file_in):
    """Loads an array of coordinates saved as .npz file. The file is expected to
    contain a numpy array named `pts`.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    FileNotFoundError
        If `file_in` is not an existing file name.
    ValueError
        If `file_in` has the wrong file extension.
    ValueError
        If the loaded array does not contain the key `pts`.
    ValueError
        If `file_in` contains a numpy array with wrong shape. The array must
        have 3 dimensions with a length of 3 for the last dimension (to ensure
        that 3D coordinates are loaded).

    Returns
    -------
    coords : ndarray, shape (N, M, 3)
        Array of 3D coordinates.

    """

    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswith(".npz"):
        raise ValueError("File has wrong format!")

    file = np.load(file_in)
    try:
        coords = file["pts"]
        if np.shape(coords)[-1] != 3 or len(np.shape(coords)) != 3:
            raise ValueError("Coordinates have wrong shape!")
    except ValueError:
        print("Loaded file does not contain the following key: pts.")
        raise

    return coords


def load_data(file_in):
    """Loads a data array saved as .npz file. The file is expected to contain a
    numpy array named `data`.

    Parameters
    ----------
    file_in : str
        File name of input file.

    Raises
    ------
    FileNotFoundError
        If `file_in` is not an existing file name.
    ValueError
        If `file_in` has the wrong file extension.
    ValueError
        If the loaded array does not contain the key `data`.
    ValueError
        If `file_in` contains a numpy array with wrong shape. The array must
        have 2 dimensions.

    Returns
    -------
    data : ndarray, shape (N, M)
        Data array.

    """

    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswith(".npz"):
        raise ValueError("File has wrong format!")

    file = np.load(file_in)
    try:
        data = file["data"]
        if len(np.shape(data)) != 2:
            raise ValueError("Data array has wrong shape!")
    except ValueError:
        print("Loaded file does not contain the following key: data.")
        raise

    return data


def save_coords(file_out, coords):
    """Saves an array of 3D coordinates as .npz file. The array is stored with
    key `pts`.

    Parameters
    ----------
    file_out : str
        File name of output file.
    coords : ndarray, shape (N, M, 3)
        Array of 3D coordinates.

    Raises
    ------
    ValueError
        If `file_out` has the wrong file extension.
    ValueError
        If `coords` has a wrong shape. The array must have 3 dimensions with a
        length of 3 for the last dimension (to ensure that 3D coordinates are
        saved).

    Returns
    -------
    None.

    """

    if not file_out.endswith(".npz"):
        raise ValueError("File has wrong format!")

    if np.shape(coords)[2] != 3 or len(np.shape(coords)) != 3:
        raise ValueError("Coordinates have wrong shape!")

    np.savez(file_out, pts=coords)


def save_data(file_out, data):
    """Saves a data array as .npz file. The array is stored with key `data`.

    Parameters
    ----------
    file_out : str
        File name of output file.
    data : ndarray, shape (N, M)
        Data array.

    Raises
    ------
    ValueError
        If `file_out` has the wrong file extension.
    ValueError
        If `data` has a wrong shape. The array must have 2 dimensions.

    Returns
    -------
    None.

    """

    if not file_out.endswith(".npz"):
        raise ValueError("File has wrong format!")

    if len(np.shape(data)) != 2:
        raise ValueError("Data array has wrong shape!")

    np.savez(file_out, data=data)


def save_meshlines(file_out, vtx1, vtx2):
    """Generates meshlines between two congruent vertex arrays and saves as
    freesurfer geometry. This visualizes point-to-point connections between both
    arrays.

    Parameters
    ----------
    file_out : str
        File name of output file.
    vtx1 : ndarray, shape (N, 3)
        First vertex array.
    vtx2 : ndarray, shape (N, 3)
        Second vertex array.

    Returns
    -------
    None.

    """

    # face of line
    fac = [[0, 1, 0]]

    vtx_res = []
    fac_res = []
    for i, j in zip(vtx1, vtx2):
        vtx_res.extend([list(i), list(j)])
        fac_res.extend(fac)
        fac[0] = [x + 2 for x in fac[0]]

    write_geometry(file_out, vtx_res, fac_res)


def coords_to_mesh(file_out, coords):
    """Generates a mesh from an array of 3D coordinates and saves as freesurfer
    geometry.

    Parameters
    ----------
    file_out : str
        File name of output file.
    coords : ndarray, shape (N, M, 3)
        Array of 3D coordinates.

    Raises
    ------
    ValueError
        If `coords` has a wrong shape. The array must have 3 dimensions with a
        length of 3 for the last dimension (to ensure that 3D coordinates are
        used).

    Returns
    -------
    None.

    """

    if np.shape(coords)[2] != 3 or len(np.shape(coords)) != 3:
        raise ValueError("Coordinates have wrong shape!")

    vtx = flatten_array(coords)
    fac = []

    length = np.prod(np.shape(coords)[:2])
    dim2 = np.shape(coords)[1]
    counter = 0
    for i in range(length - dim2):
        if not np.mod(counter, dim2 - 1) and counter != 0:
            counter = 0
            continue
        else:
            counter += 1

            fac.append([i, i + 1, i + dim2])
            fac.append([i + dim2 + 1, i + dim2, i + 1])

    fac = np.array(fac)
    write_geometry(file_out, vtx, fac)


def data_to_overlay(file_out, data):
    """Generates an overlay from a data array and saves as .mgh file.

    Parameters
    ----------
    file_out : str
        File name of output file.
    data : ndarray, shape (N, M)
        Data array.

    Raises
    ------
    ValueError
        If `file_out` has the wrong file extension.
    ValueError
        If `data` has a wrong shape. The array must have 2 dimensions.

    Returns
    -------
    None.

    """

    if not file_out.endswith(".mgh"):
        raise ValueError("File has wrong format!")

    if len(np.shape(data)) != 2:
        raise ValueError("Data array has wrong shape!")

    out = flatten_array(data)
    save_overlay(file_out, out)

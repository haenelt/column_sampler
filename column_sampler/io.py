# -*- coding: utf-8 -*-
"""I/O functions."""

import os
import numpy as np
from nibabel.freesurfer.io import write_geometry
from column_filter.io import save_overlay
from util import flatten_array

__all__ = ["read_anchor", "load_coords", "load_data", "save_coords",
           "save_meshlines", "save_data", "coords_to_mesh", "data_to_overlay"]


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
            res[str(i)] = [int(e) if e.isdigit() else e
                           for e in tmp.split(',')]

    return res


def load_coords(file_in):
    """Loads an array of coordinates saved as .npz file. The file is expected
    to contain a numpy array named `pts`.

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
        If `file_in` contains a numpy array with wrong shape. The last array
        dimension must have length 3 to ensure that 3D coordinates are loaded.

    Returns
    -------
    coords : (N, ..., 3) np.ndarray
        Array of 3D coordinates.

    """

    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswith(".npz"):
        raise ValueError("File has wrong format!")

    file = np.load(file_in)
    try:
        coords = file["pts"]
        if not np.shape(coords)[-1] == 3:
            raise ValueError("Coordinates have wrong shape!")
    except ValueError:
        print("Loaded file does not contain the following key: pts.")

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

    Returns
    -------
    coords : (N, ...) np.ndarray
        Data array.

    """

    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswidth(".npz"):
        raise ValueError("File has wrong format!")

    file = np.load(file_in)
    try:
        data = file["data"]
    except ValueError:
        print("Loaded file does not contain the following key: data.")

    return data


def save_coords(file_out, coords):
    """Saves an array of 3D coordinates as .npz file. The array is stored with
    key `pts`.

    Parameters
    ----------
    file_out : str
        File name of output file.
    coords : (N, ..., 3) np.ndarray
        Array of 3D coordinates.

    Raises
    ------
    ValueError
        If `file_in` has the wrong file extension.
    ValueError
        If `coords` has a wrong shape. The last array dimension must have
        length 3 to ensure that 3D coordinates are saved.

    Returns
    -------
    None.

    """

    if not file_out.endswidth(".npz"):
        raise ValueError("File has wrong format!")

    if not np.shape(coords)[-1] == 3:
        raise ValueError("Coordinates have wrong shape!")

    np.savez(file_out, pts=coords)


def save_meshlines(file_out, v1, v2):
    """

    This function returns a vertex and a corresponding face array to visualize
    point-to-point connections between two congruent surface meshs.

    Parameters
    ----------
    file_out : str
        File name of output file.
    v1
    v2

    Returns
    -------

    """

    # face of line
    fac = [[0, 1, 0]]

    vtx_res = []
    fac_res = []
    for i, j in zip(v1, v2):
        vtx_res.extend([list(i), list(j)])
        fac_res.extend(fac)
        fac[0] = [x + 2 for x in fac[0]]

    write_geometry(file_out, vtx_res, fac_res)


def save_data(file_out, data):
    if not file_out.endswidth(".npz"):
        raise ValueError("File has wrong format!")

    return np.savez(file_out, data=data)


def coords_to_mesh(file_out, coords):
    second_dim = np.shape(coords)[1]
    yyy = flatten_array(coords)
    faces = []
    counter1 = 0
    length = np.prod(np.shape(coords)[:2])
    for i in range(length - second_dim):
        if not np.mod(counter1, second_dim - 1) and counter1 != 0:
            counter1 = 0
            continue
        else:
            counter1 += 1

            faces.append([i, i + 1, i + second_dim])
            faces.append([i + second_dim + 1, i + second_dim, i + 1])
    faces = np.array(faces)
    write_geometry(file_out, yyy, faces)


def data_to_overlay(file_out, data):
    if not file_out.endswidth(".mgh"):
        raise ValueError("File has wrong format!")

    out = flatten_array(data)
    save_overlay(file_out, out)

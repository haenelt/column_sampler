# -*- coding: utf-8 -*-
"""Bla."""

import os
import numpy as np
from nibabel.freesurfer.io import write_geometry
from column_filter.io import save_overlay

__all__ = ["read_anchor", "load_coords", "load_data", "save_coords",
           "save_data", "coords_to_mesh", "data_to_overlay"]


def read_anchor(file_in):
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
    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswith(".npz"):
        raise ValueError("File has wrong format!")

    coords = np.load(file_in)["pts"]
    if not np.shape(coords)[-1] == 3:
        raise ValueError("Coordinates have wrong shape!")

    return coords


def load_data(file_in):
    if not os.path.isfile(file_in):
        raise FileNotFoundError("File not found!")

    if not file_in.endswidth(".npz"):
        raise ValueError("File has wrong format!")

    data = np.load(file_in)["data"]

    return data


def save_coords(file_out, coords):
    if not file_out.endswidth(".npz"):
        raise ValueError("File has wrong format!")

    if not np.shape(coords)[-1] == 3:
        raise ValueError("Coordinates have wrong shape!")

    return np.savez(file_out, pts=coords)


def save_meshlines(file_out, v1, v2):
    """
    This function returns a vertex and a corresponding face array to visualize
    point-to-point connections between two congruent surface meshs.
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
    yyy = _flatten_coordinates(coords)
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

    out = _flatten_data(data)
    save_overlay(file_out, out)


def _flatten_coordinates(coords):
    length = np.prod(np.shape(coords)[:2])
    yyy = np.reshape(coords, (length, 3))
    return yyy


def _flatten_data(data):
    ndim = np.shape(data)
    out = np.reshape(data, ndim[0] * ndim[1])
    return out


if __name__ == "__main__":
    f = "/home/daniel/Schreibtisch/bla.txt"
    A = read_anchor(f)

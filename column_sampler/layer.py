# -*- coding: utf-8 -*-
"""Bla."""

import functools
import numpy as np

__all__ = ["Layer"]


class Layer:
    def __init__(self, coords, vtx_ref, fac_ref, vtx_white, vtx_pial):
        self.coords = coords  # line x pt x coords
        self.vtx_ref = vtx_ref
        self.fac_ref = fac_ref
        self.vtx_white = vtx_white
        self.vtx_pial = vtx_pial

    @property
    @functools.lru_cache
    def closest_face(self):
        coords_flat = self._flatten_array(self.coords)
        x_flat = coords_flat[:, 0]
        y_flat = coords_flat[:, 1]
        z_flat = coords_flat[:, 2]

        # only for first face
        arr = np.zeros((len(coords_flat), len(self.fac_ref)))

        for i in range(3):
            arr += (x_flat[:, None] - self.vtx_ref[self.fac_ref[:, i], 0]) ** 2
            arr += (y_flat[:, None] - self.vtx_ref[self.fac_ref[:, i], 1]) ** 2
            arr += (z_flat[:, None] - self.vtx_ref[self.fac_ref[:, i], 2]) ** 2

        arr = np.sqrt(arr)
        ind_fac = np.argmin(arr, axis=1)
        ind_fac = self._unflatten_array(ind_fac, self.coords[:, :, 0])

        return ind_fac

    @property
    @functools.lru_cache
    def border_points(self):
        coords_flat = self._flatten_array(self.coords)
        fac_flat = self._flatten_array(self.closest_face)

        pial = np.zeros((len(fac_flat), 3))
        white = np.zeros((len(fac_flat), 3))
        for i in range(3):
            pial += self.vtx_pial[self.fac_ref[fac_flat, i]]
            white += self.vtx_white[self.fac_ref[fac_flat, i]]
        pial /= 3
        white /= 3

        x = pial - white
        x_pial_dist = np.linalg.norm(coords_flat - pial, axis=1)
        x_white_dist = np.linalg.norm(coords_flat - white, axis=1)

        pt1 = coords_flat.copy()
        pt2 = coords_flat + x

        pt1_moved = [self._line_equation(-x, pt1[i], pt2[i])
                     for i, x in enumerate(x_white_dist)]
        pt2_moved = [self._line_equation(x, pt1[i], pt2[i])
                     for i, x in enumerate(x_pial_dist)]
        pt1_moved = self._unflatten_array(pt1_moved, self.coords)
        pt2_moved = self._unflatten_array(pt2_moved, self.coords)

        return pt1_moved, pt2_moved

    def generate_layer(self, n_layer):
        w, p = self.border_points
        dim1, dim2 = np.shape(w)[:2]
        layer = np.zeros((n_layer, dim1, dim2, 3))
        for i in range(n_layer):
            vtx_layer = w + i / (n_layer - 1) * (p - w)
            layer[i, :, :, :] = vtx_layer

        return layer

    @staticmethod
    def _line_equation(x, a, b):
        return a + x * (b - a) / np.linalg.norm(a - b)

    @staticmethod
    def _flatten_array(pts):
        pts = np.array(pts)
        dim1, dim2 = np.shape(pts)[:2]
        if np.shape(pts)[-1] == 3 and len(np.shape(pts)) > 2:
            return np.reshape(pts, (dim1 * dim2, 3))
        else:
            return np.reshape(pts, (dim1 * dim2))

    @staticmethod
    def _unflatten_array(pts, pts_ref):
        pts = np.array(pts)
        pts_ref = np.array(pts_ref)
        dim1, dim2 = np.shape(pts_ref)[:2]
        if np.shape(pts)[-1] == 3 and len(np.shape(pts)) > 2:
            return np.reshape(pts, (dim1, dim2, 3))
        else:
            return np.reshape(pts, (dim1, dim2))


if __name__ == "__main__":
    from nibabel.freesurfer.io import read_geometry
    from column_sampler.io import load_coords

    # filenames
    file_white = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_0"
    file_middle = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
    file_pial = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_10"
    file_coords = "/home/daniel/Schreibtisch/bb100.npz"

    # load vertices and faces
    v_ref, f_ref = read_geometry(file_middle)
    v_white, _ = read_geometry(file_white)
    v_pial, _ = read_geometry(file_pial)
    cds = load_coords(file_coords)

    A = Layer(cds, v_ref, f_ref, v_white, v_pial)
    bla = A.generate_layer(11)

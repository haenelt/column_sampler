# -*- coding: utf-8 -*-
"""Cortical layer definition."""

import functools
import numpy as np
from util import flatten_array, unflatten_array

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
        coords_flat = flatten_array(self.coords)
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
        ind_fac = unflatten_array(ind_fac, self.coords)

        return ind_fac

    @property
    @functools.lru_cache
    def border_coordinates(self):
        coords_flat = flatten_array(self.coords)
        fac_flat = flatten_array(self.closest_face)

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

        pt1_moved = [self._line_point(-x, pt1[i], pt2[i])
                     for i, x in enumerate(x_white_dist)]
        pt2_moved = [self._line_point(x, pt1[i], pt2[i])
                     for i, x in enumerate(x_pial_dist)]
        pt1_moved = unflatten_array(pt1_moved, self.coords)
        pt2_moved = unflatten_array(pt2_moved, self.coords)

        return pt1_moved, pt2_moved

    def generate_layer(self, n_layer):
        w, p = self.border_coordinates
        dim1, dim2 = np.shape(w)[:2]
        layer = np.zeros((n_layer, dim1, dim2, 3))
        for i in range(n_layer):
            vtx_layer = w + i / (n_layer - 1) * (p - w)
            layer[i, :, :, :] = vtx_layer

        return layer

    @staticmethod
    def _line_point(x, a, b):
        return a + x * (b - a) / np.linalg.norm(a - b)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, c):
        c = np.asarray(c)
        if c.ndim != 3 or np.shape(c)[2] != 3:
            raise ValueError("Coordinates have wrong shape!")

        self._coords = c

    @property
    def vtx_ref(self):
        return self._vtx_ref

    @vtx_ref.setter
    def vtx_ref(self, v):
        v = np.asarray(v)
        if v.ndim != 2 or np.shape(v)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        self._vtx_ref = v

    @property
    def fac_ref(self):
        return self._fac_ref

    @fac_ref.setter
    def fac_ref(self, f):
        f = np.asarray(f)
        if f.ndim != 2 or np.shape(f)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        if np.max(f) != len(self.vtx_ref) - 1:
            raise ValueError("Faces do not match vertex array!")

        self._fac_ref = f

    @property
    def vtx_white(self):
        return self._vtx_white

    @vtx_white.setter
    def vtx_white(self, v):
        v = np.asarray(v)
        if v.ndim != 2 or np.shape(v)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        self._vtx_white = v

    @property
    def vtx_pial(self):
        return self._vtx_pial

    @vtx_pial.setter
    def vtx_pial(self, v):
        v = np.asarray(v)
        if v.ndim != 2 or np.shape(v)[1] != 3:
            raise ValueError("Vertices have wrong shape!")

        self._vtx_pial = v

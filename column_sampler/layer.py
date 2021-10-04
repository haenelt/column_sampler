# -*- coding: utf-8 -*-
"""Cortical layer definition."""

import functools
import numpy as np
from util import flatten_array, unflatten_array

__all__ = ["Layer"]


class Layer:
    """Layer computation.

    Definition of equi-distance layers between white and pial surface for an
    array of 3D coordinate.

    Parameters
    ----------
    coords : np.ndarray, shape=(N, M, 3)
        Coordinate array for N lines containing M 3D line coordinates.
    vtx_ref : np.ndarray, shape=(N, 3)
        Vertex array of reference surface.
    fac_ref : np.ndarray, shape=(M, 3)
        Corresponding face array.
    vtx_white : np.ndarray, shape=(N, 3)
        Vertex array of white surface.
    vtx_pial : np.ndarray, shape=(N, 3)
        Vertex array of pial surface.

    Raises
    ------
    ValueError :
        If `coords` has an invalid shape.
    ValueError :
        If `vtx_ref` has an invalid shape.
    ValueError :
        If `fac_ref` has an invalid shape or does not match the vertex array
        `vtx_ref`.
    ValueError :
        If `vtx_white` has an invalid shape.
    ValueError :
        If `vtx_pial` has an invalid shape.

    """

    def __init__(self, coords, vtx_ref, fac_ref, vtx_white, vtx_pial):
        self.coords = coords  # line x pt x coords
        self.vtx_ref = vtx_ref
        self.fac_ref = fac_ref
        self.vtx_white = vtx_white
        self.vtx_pial = vtx_pial

    @property
    @functools.lru_cache
    def closest_face(self):
        """Finds the closest face defined by euclidean distance for each point
        in the coordinate array.

        Returns
        -------
        ind_fac : np.ndarray, shape=(N, M, 3)
            Array of face indices for closest faces to array coordinates.

        """

        coords_flat = flatten_array(self.coords)
        x_flat = coords_flat[:, 0]
        y_flat = coords_flat[:, 1]
        z_flat = coords_flat[:, 2]

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
        """Computes corresponding white and pial coordinates for the input
        coordinate array.

        Returns
        -------
        pt_white : np.ndarray, shape=(N, M, 3)
            Array of coordinates shifted to white surface.
        pt_pial : np.ndarray, shape=(N, M, 3)
            Array of coordinates shifted to pial surface.

        """

        coords_flat = flatten_array(self.coords)
        fac_flat = flatten_array(self.closest_face)

        # average white and pial vertex coordinates
        pial = np.zeros((len(fac_flat), 3))
        white = np.zeros((len(fac_flat), 3))
        for i in range(3):
            pial += self.vtx_pial[self.fac_ref[fac_flat, i]]
            white += self.vtx_white[self.fac_ref[fac_flat, i]]
        pial /= 3
        white /= 3

        # distance of coordinate array to white and pial coordinates
        x = pial - white
        x_pial_dist = np.linalg.norm(coords_flat - pial, axis=1)
        x_white_dist = np.linalg.norm(coords_flat - white, axis=1)

        # shift each coordinate along meshlines to white and pial surface
        pt1 = coords_flat.copy()
        pt2 = coords_flat + x

        pt_white = np.array([self._line_point(-x, pt1[i], pt2[i])
                             for i, x in enumerate(x_white_dist)])
        pt_pial = np.array([self._line_point(x, pt1[i], pt2[i])
                            for i, x in enumerate(x_pial_dist)])
        pt_white = unflatten_array(pt_white, self.coords)
        pt_pial = unflatten_array(pt_pial, self.coords)

        return pt_white, pt_pial

    def generate_layers(self, n_layer):
        """Defines equi-distant layers between white and pial surface
        coordinates computed from the input coordinate array.

        Parameters
        ----------
        n_layer : int
            Number of layers.

        Returns
        -------
        layer : np.ndarray(L, N, M, 3)
            Array of coordinates for L layers.

        """

        w, p = self.border_coordinates
        dim1, dim2 = np.shape(w)[:2]
        layer = np.zeros((n_layer, dim1, dim2, 3))
        for i in range(n_layer):
            vtx_layer = w + i / (n_layer - 1) * (p - w)
            layer[i, :, :, :] = vtx_layer

        return layer

    @staticmethod
    def _line_point(t, pts1, pts2):
        """Defines a line between two points and returns the line coordinates at
        position t.

        Parameters
        ----------
        t : float
            Line position for which coordinates are computed.
        pts1 : np.ndarray, shape=(3,)
            First line point.
        pts2 : np.ndarray, shape=(3,)
            Second line point.

        Returns
        -------
        np.ndarray, shape=(3,)
            Line coordinates at position t.

        """

        return pts1 + t * (pts2 - pts1) / np.linalg.norm(pts1 - pts2)

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
            raise ValueError("Faces have wrong shape!")

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

# -*- coding: utf-8 -*-
"""Bla."""

import os
import functools
import numpy as np

__all__ = ["Layer"]


class Layer:
    N_LAYER = 11

    def __init__(self, coords, vtx_ref, fac_ref, vtx_white, vtx_pial):
        self.coords = coords  # line x pt x coords
        self.vtx_ref = vtx_ref
        self.fac_ref = fac_ref
        self.vtx_white = vtx_white
        self.vtx_pial = vtx_pial

    @property
    @functools.lru_cache
    def closest_face(self):
        dim1, dim2 = np.shape(self.coords)[:2]
        coords_flat = np.reshape(self.coords, (dim1 * dim2, 3))

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
        self.ind_fac = np.argmin(arr, axis=1)
        self.ind_fac = np.reshape(self.ind_fac, (dim1, dim2))

        return self.ind_fac

    @property
    @functools.lru_cache
    def border_points(self):
        fac = self.closest_face
        dim1, dim2 = np.shape(fac)
        coords_flat = np.reshape(self.coords, (dim1*dim2, 3))
        fac_flat = np.reshape(fac, dim1*dim2)

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
        self.pt1_moved = [self._line_equation(-x, pt1[i], pt2[i]) for i, x in enumerate(x_white_dist)]
        self.pt2_moved = [self._line_equation(x, pt1[i], pt2[i]) for i, x in enumerate(x_pial_dist)]
        self.pt1_moved = np.array(self.pt1_moved)
        self.pt2_moved = np.array(self.pt2_moved)

        mesh = self.meshlines2(self.pt1_moved, self.pt2_moved)
        write_geometry("/home/daniel/Schreibtisch/test", mesh[0], mesh[1])

        self.pt1_moved = np.reshape(self.pt1_moved, (dim1, dim2, 3))
        self.pt2_moved = np.reshape(self.pt2_moved, (dim1, dim2, 3))

        return self.pt1_moved, self.pt2_moved

    def layer(self):
        w, p = self.border_points
        dim1, dim2 = np.shape(w)[:2]
        bla = np.zeros((self.N_LAYER, dim1, dim2, 3))
        for i in range(self.N_LAYER):
            vtx_layer = w + i / (self.N_LAYER - 1) * (p - w)
            bla[i, :, :, :] = vtx_layer

        return bla

    @staticmethod
    def _line_equation(x, a, b):
        return a + x * (b - a) / np.linalg.norm(a - b)

    @staticmethod
    def meshlines2(v1, v2):
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

        return np.array(vtx_res), np.array(fac_res)

    @property
    def meshlines(self):
        """
        This function returns a vertex and a corresponding face array to visualize
        point-to-point connections between two congruent surface meshs.
        """

        # face of line
        fac = [[0, 1, 0]]

        vtx_res = []
        fac_res = []
        for i, j in zip(self.vtx_white, self.vtx_pial):
            vtx_res.extend([list(i), list(j)])
            fac_res.extend(fac)
            fac[0] = [x + 2 for x in fac[0]]

        return np.array(vtx_res), np.array(fac_res)


if __name__ == "__main__":
    from nibabel.freesurfer.io import read_geometry, write_geometry
    from column_sampler.io import load_coords, coords_to_mesh

    # filenames
    file_white = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_0"
    file_middle = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
    file_pial = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_10"
    file_coords = "/home/daniel/Schreibtisch/bb100.npz"

    # load vertices and faces
    vtx_ref, fac_ref = read_geometry(file_middle)
    vtx_white, _ = read_geometry(file_white)
    vtx_pial, _ = read_geometry(file_pial)
    coords = load_coords(file_coords)

    A = Layer(coords, vtx_ref, fac_ref, vtx_white, vtx_pial)
    bla = A.layer()

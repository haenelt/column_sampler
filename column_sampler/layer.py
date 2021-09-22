# -*- coding: utf-8 -*-
"""Bla."""

import numpy as np

__all__ = ["Layer"]


class Layer:
    def __init__(self, coords, vtx_ref, fac_ref, vtx_white, vtx_pial):
        self.coords = coords
        self.vtx_ref = vtx_ref
        self.fac_ref = fac_ref
        self.vtx_white = vtx_white
        self.vtx_pial = vtx_pial

    # for each vertex in curved mesh
    # get mean distance to face vertices
    # get mean meshline from meshlines of face
    # divide into layers
    # output layer coordinates

    def get_distance(self):
        a = np.array(1,1,1)
        b = fac[]
        np.linalg.norm(a-b)

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

    file_in1 = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_0"
    file_in2 = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_10"

    vtx1, _ = read_geometry(file_in1)
    vtx2, _ = read_geometry(file_in2)

    A = Layer(vtx1, vtx2)
    B = A.meshlines()

    write_geometry("/home/daniel/Schreibtisch/bla", B[0], B[1])

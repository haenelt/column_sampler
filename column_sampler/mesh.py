# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
import nibabel as nb
import networkx as nx
from numpy.linalg import norm
from column_filter import mesh
from nibabel.affines import apply_affine
from networkx.algorithms.shortest_paths.generic import shortest_path

__all__ = ["Mesh"]


class Mesh(mesh.Mesh):

    def __init__(self, vtx, fac):
        super().__init__(vtx, fac)

    @property
    @functools.lru_cache
    def iter_edges(self):
        for a, b, c in self.fac:
            yield a, b, self.euclidean_distance(self.vtx[a], self.vtx[b])
            yield b, c, self.euclidean_distance(self.vtx[b], self.vtx[c])
            yield a, c, self.euclidean_distance(self.vtx[a], self.vtx[c])

    def ras2vox(self, vol_in):
        """
        https://neurostars.org/t/get-voxel-to-ras-transformation-from-nifti-file/4549
        """

        nii = nb.load(vol_in)
        mgh = nb.MGHImage(nii.dataobj, nii.affine)

        vox2ras_tkr = mgh.header.get_vox2ras_tkr()
        ras2vox_tkr = np.linalg.inv(vox2ras_tkr)

        return apply_affine(ras2vox_tkr, self.vtx)  # vox2ras_tkr

    def path_dijkstra(self, ind):
        """
        https://gallantlab.github.io/pycortex/_modules/cortex/freesurfer.html
        """
        graph = nx.Graph()
        graph.add_weighted_edges_from(self.iter_edges)
        path = []
        for i in range(len(ind)-1):
            tmp = shortest_path(graph,
                                source=ind[i],
                                target=ind[i+1],
                                weight='weight',
                                method='dijkstra')
            path.extend(tmp[:-1])
            path.append(ind[-1])

        return path

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(pt2 - pt1)


class ColumnMesh(Mesh):
    def __init__(self, vtx, fac, ind):
        super().__init__(vtx, fac)
        self.ind = ind

    def perpendicular_line(self, line_length=1, line_step=0.1, ndir_smooth=5):

        # line equation
        y = lambda x, a, b: np.array([a + x * (a - b) / norm(a - b) for x in x])

        # get surface normals
        normal = self.vertex_normals

        # initialize line coordinate parameters
        x = np.arange(-line_length, line_length + line_step, line_step)

        # initialize list
        line_temp = []

        # get perpendicular line for each line vertex
        # for j in range(ndir_smooth,len(line)-ndir_smooth):
        for j in range(2, len(self.ind) - 2):

            # get local neighborhood
            nn = self.neighborhood(self.ind[j])

            # For a stable normal line computation, neighbor points in
            # ndir_smooth distance are selected. This prevents the use of points
            # at both line endings. For those cases, the neighbor distance is
            # shortened.
            if j < ndir_smooth:
                p01 = np.mean(self.vtx[self.ind[j + 1:j + ndir_smooth], :],
                              axis=0) - self.vtx[self.ind[j], :]
                p02 = np.mean(self.vtx[self.ind[0:j - 1], :], axis=0) - self.vtx[self.ind[j], :]
            elif j > len(self.ind) - ndir_smooth:
                p01 = np.mean(self.vtx[self.ind[j + 1:len(self.ind)], :], axis=0) - self.vtx[self.ind[j],:]
                p02 = np.mean(self.vtx[self.ind[j - ndir_smooth:j - 1], :], axis=0) - self.vtx[self.ind[j], :]
            else:
                p01 = np.mean(self.vtx[self.ind[j + 1:j + ndir_smooth], :],
                              axis=0) - self.vtx[self.ind[j], :]
                p02 = np.mean(self.vtx[self.ind[j - ndir_smooth:j - 1], :],
                              axis=0) - self.vtx[self.ind[j], :]

            # get smoothed surface normal
            p1 = np.mean(normal[nn], axis=0)

            # get mean vector normal to line and surface normal direction
            p21 = np.cross(p01, p1)
            p22 = np.cross(p1, p02)  # flip for consistent line direction
            p2 = (p21 + p22) / 2

            # add current line vertex point
            p0 = self.vtx[self.ind[j], :]
            p2 += self.vtx[self.ind[j], :]

            # get coordinates of perpendicular line
            yy = y(x, p0, p2)

            # append to list
            line_temp.append(yy)

        return line_temp




"""
self.vtx = [vtx_smooth[j] - np.dot(np.outer(v, v), vtx_diff[j])
                        for j, v in enumerate(n)]
"""


















from nibabel.freesurfer.io import read_geometry
surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
ind = [159488, 30645, 171581]
vtx, fac = read_geometry(surf_in)
A = Mesh(vtx, fac)
res = A.path_dijkstra(ind)

B = ColumnMesh(vtx, fac, res)
xxx = B.perpendicular_line()

array_dims = np.shape(xxx)
yyy = np.reshape(xxx, (np.shape(xxx)[0]*np.shape(xxx)[1],3))

import pyvista as pv
from nibabel.freesurfer.io import write_geometry
cloud = pv.PolyData(yyy)
surf = cloud.delaunay_2d(alpha=0.5)

faces = np.reshape(surf.faces, (int(len(surf.faces)/4),4))
faces = faces[:,1:]

write_geometry("/home/daniel/Schreibtisch/bb2", yyy, faces)
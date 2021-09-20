# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import networkx as nx
from numpy.linalg import norm
from column_filter import mesh
from networkx.algorithms.shortest_paths.generic import shortest_path
import pyvista as pv
from nibabel.freesurfer.io import write_geometry
import numpy as np
from column_sampler.util import sample_line
from numpy.linalg import norm
import matplotlib.pyplot as plt

__all__ = ["PlanarMesh"]


class PlanarMesh(mesh.Mesh):
    def __init__(self, vtx, fac, ind):
        super().__init__(vtx, fac)
        self.ind = ind
        self.line = None
        self.point_cloud = None
        self.data = None

    @property
    def path_dijkstra(self):
        """
        https://gallantlab.github.io/pycortex/_modules/cortex/freesurfer.html
        """
        graph = nx.Graph()
        graph.add_weighted_edges_from(self._iter_edges)
        self.line = []
        for i in range(len(ind)-1):
            tmp = shortest_path(graph,
                                source=self.ind[i],
                                target=self.ind[i+1],
                                weight='weight',
                                method='dijkstra')
            self.line.extend(tmp[:-1])
        self.line.append(ind[-1])

        return self.line

    def perpendicular_line(self, line_length=1, line_step=0.1, ndir_smooth=5):

        # line equation
        y = lambda x, a, b: np.array([a + x * (a - b) / norm(a - b) for x in x])

        # get surface normals
        normal = self.vertex_normals

        # initialize line coordinate parameters
        x = np.arange(-line_length, line_length + line_step, line_step)

        # check line
        if not self.line:
            raise ValueError("You did not run dijkstra")

        # initialize list
        self.point_cloud = []

        def _mean(pts):
            return np.mean(pts, axis=0)

        # get perpendicular line for each line vertex
        ind_len = len(self.line)
        for j in range(2, ind_len - 2):

            # get local neighborhood
            nn = self.neighborhood(self.line[j])
            p0 = self.vtx[self.line[j], :]
            p = self.vtx[self.line] - p0

            j_next = j + ndir_smooth
            j_prev = j - ndir_smooth

            # For a stable normal line computation, neighbor points in
            # ndir_smooth distance are selected. This prevents the use of points
            # at both line endings. For those cases, the neighbor distance is
            # shortened.
            if j < ndir_smooth:
                p01 = _mean(p[j + 1:j_next, :])
                p02 = _mean(p[0:j - 1, :])
            elif j > ind_len - ndir_smooth:
                p01 = _mean(p[j + 1:ind_len, :])
                p02 = _mean(p[j_prev:j - 1, :])
            else:
                p01 = _mean(p[j + 1:j_next, :])
                p02 = _mean(p[j_prev:j - 1, :])

            #p01 = _mean(p[j + 1:, :])
            #p02 = _mean(p[0:j - 1, :])


            # get smoothed surface normal
            p1 = np.mean(normal[nn], axis=0)

            # get mean vector normal to line and surface normal direction
            p21 = np.cross(p01, p1)
            p22 = np.cross(p1, p02)  # flip for consistent line direction
            p2 = (p21 + p22) / 2

            # add current line vertex point
            p2 += p0

            # get coordinates of perpendicular line
            yy = y(x, p0, p2)

            # append to list
            self.point_cloud.append(yy)

        return self.point_cloud

    def sample_data(self, vol_in, deform_in):

        self.data = []
        for i in range(len(self.point_cloud)):
            tmp = sample_line(self.point_cloud[i], vol_in, deform_in)
            self.data.append(tmp)

        return self.data

    def plot_data(self):
        for i in range(len(self.data)):
            plt.plot(self.data[i])
        plt.show()

    def save_line(self):
        from column_filter.io import save_overlay
        out = np.zeros(len(self.vtx))
        out[self.line] = 1
        save_overlay("/home/daniel/Schreibtisch/line.mgh", out)

    def save_points(self):
        pass

    def save_overlay(self):
        ndims = np.shape(self.data)
        out = np.reshape(self.data, ndims[0] * ndims[1])
        from column_filter.io import save_overlay
        save_overlay("/home/daniel/Schreibtisch/bbb.mgh", out)

    def save_mesh(self, file_out, alpha=0.5):

        first_dim = np.shape(self.point_cloud)[0]
        second_dim = np.shape(self.point_cloud)[1]

        yyy = self._flatten_coordinates()
        cloud = pv.PolyData(yyy)
        surf = cloud.delaunay_2d(alpha=alpha)

        faces = []
        counter1 = 0
        for i in range(len(yyy)):
            faces.append([i, i+1, i + second_dim])
        faces = np.array(faces)

        #faces = np.reshape(surf.faces, (int(len(surf.faces) / 4), 4))
        #faces = faces[:, 1:]

        write_geometry(file_out, yyy, faces)

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(pt2 - pt1)

    def _flatten_coordinates(self):
        array_dims = np.shape(self.point_cloud)
        yyy = np.reshape(self.point_cloud, (array_dims[0] * array_dims[1], 3))
        return yyy

    @property
    @functools.lru_cache
    def _iter_edges(self):
        for a, b, c in self.fac:
            yield a, b, self.euclidean_distance(self.vtx[a], self.vtx[b])
            yield b, c, self.euclidean_distance(self.vtx[b], self.vtx[c])
            yield a, c, self.euclidean_distance(self.vtx[a], self.vtx[c])



























































from nibabel.freesurfer.io import read_geometry
surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
#surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.white_match_final_inflate30"
surf_out = "/home/daniel/Schreibtisch/bb100"
#ind = [40659, 189512, 181972]
ind = [98897, 104039, 112792, 112851, 113110, 101984]
v, f = read_geometry(surf_in)
vol_in = "/home/daniel/Schreibtisch/data/data_sampler/vol/Z_all_left_right_GE_EPI3.nii"
deform_in = "/home/daniel/Schreibtisch/data/data_sampler/vol/target2source.nii.gz"

A = PlanarMesh(v, f, ind)

x1 = A.path_dijkstra
x2 = A.perpendicular_line()
x3 = A.sample_data(vol_in, deform_in)
x4 = A.plot_data()
A.save_mesh(surf_out)
A.save_line()
A.save_overlay()

















first_dim = np.shape(A.point_cloud)[0]
second_dim = np.shape(A.point_cloud)[1]

yyy = A._flatten_coordinates()

faces = []
counter1 = 0
counter2 = 0
for i in range(len(yyy)-second_dim):

    if not np.mod(counter1, second_dim-1) and counter1 != 0:
        print(i)
        counter1 = 0
        continue
    else:
        counter1 += 1


        faces.append([i, i + 1, i + second_dim])
        faces.append([i+second_dim+1, i + second_dim,i+1])
faces = np.array(faces)

write_geometry("/home/daniel/Schreibtisch/aber", yyy, faces)

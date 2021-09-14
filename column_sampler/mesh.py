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
import pyvista as pv
from nibabel.freesurfer.io import write_geometry

__all__ = ["LineMesh", "ColumnMesh"]


class LineMesh(mesh.Mesh):
    def __init__(self, vtx, fac, ind):
        super().__init__(vtx, fac)
        self.ind = ind


class ColumnMesh(mesh.Mesh):
    def __init__(self, vtx, fac, ind):
        super().__init__(vtx, fac)
        self.ind = self.path_dijkstra(ind)
        self.line_temp = []

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

    def perpendicular_line(self, line_length=1, line_step=0.1, ndir_smooth=5):

        # line equation
        y = lambda x, a, b: np.array([a + x * (a - b) / norm(a - b) for x in x])

        # get surface normals
        normal = self.vertex_normals

        # initialize line coordinate parameters
        x = np.arange(-line_length, line_length + line_step, line_step)

        # initialize list
        self.line_temp = []

        def _mean(pts):
            return np.mean(pts, axis=0)

        # get perpendicular line for each line vertex
        # for j in range(ndir_smooth,len(line)-ndir_smooth):
        ind_len = len(self.ind)
        for j in range(2, ind_len - 2):

            # get local neighborhood
            nn = self.neighborhood(self.ind[j])
            p0 = self.vtx[self.ind[j], :]
            p = self.vtx[self.ind] - p0

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

            yy = self.remesh(yy)

            # append to list
            self.line_temp.append(yy)

        return self.line_temp

    def closest_point(self, pt):

        vtx_tmp = self.vtx - pt
        dist = np.sqrt(vtx_tmp[:, 0]**2+vtx_tmp[:, 1]**2+vtx_tmp[:, 2]**2)
        ind = np.where(dist == np.min(dist))[0][0]

        return ind

    def remesh(self, pts):
        res = []
        for bla in pts:
            ind_here = self.closest_point(bla)
            n = self.vertex_normals[ind_here, :]

            # get closest vertex
            # get normal
            # for each y -> remesh with formula

            res.append(bla-np.dot(np.outer(n, n), bla-self.vtx[ind_here,:]))

        return res

    def update_coordinates_proc(self, axis=[0, 1]):
        for i in axis:
            print(i)
            self.update_coordinates(i)

        return self.line_temp

    def update_coordinates(self, axis=0):

        # pts -> lines x pts x coords
        pts = self.line_temp.copy()
        counter = 0
        while counter < 100000:
            counter += 1
            #print(counter)

            # random line point
            if axis == 0:
                x_random = np.random.randint(1, np.shape(pts)[0]-1)
                y_random = np.random.randint(np.shape(pts)[1])

                p = pts[x_random][y_random]
                p_prev = pts[x_random-1][y_random]
                p_next = pts[x_random+1][y_random]
            elif axis == 1:
                x_random = np.random.randint(np.shape(pts)[0])
                y_random = np.random.randint(1, np.shape(pts)[1]-1)

                p = pts[x_random][y_random]
                p_prev = pts[x_random][y_random-1]
                p_next = pts[x_random][y_random+1]
            else:
                raise ValueError("Invalid argument for axis!")

            dist_prev = self.euclidean_distance(p, p_prev)
            dist_next = self.euclidean_distance(p, p_next)
            if dist_next / dist_prev > 0.1:

                # new line point coordinates is the mean of neighboring
                # coordinates
                pts[x_random][y_random][0] = (p_prev[0] + p_next[0]) / 2
                pts[x_random][y_random][1] = (p_prev[1] + p_next[1]) / 2
                pts[x_random][y_random][2] = (p_prev[2] + p_next[2]) / 2

            # remesh all line coordinates
            if not np.mod(counter, 10000):
                for i, line in enumerate(pts):
                    pts[i] = self.remesh(line)

            if not np.mod(counter, 1000):
                cost = self.check_homogeneity(pts, axis=axis)
                #print(cost)
                if cost < 1e-4:
                    for i, line in enumerate(pts):
                        pts[i] = self.remesh(line)
                    #print(counter)
                    break

        self.line_temp = pts.copy()

        return pts

    def check_homogeneity(self, arr, axis=0):
        dist = []
        for i in range(1,np.shape(arr)[0]-1):
            for j in range(1,np.shape(arr)[1]-1):

                p = arr[i][j]

                # random line point
                if axis == 0:
                    p_prev = arr[i - 1][j]
                    p_next = arr[i + 1][j]
                elif axis == 1:
                    p_prev = arr[i][j- 1]
                    p_next = arr[i][j + 1]
                else:
                    raise ValueError("Invalid argument for axis!")

                dist_prev = self.euclidean_distance(p, p_prev)
                dist_next = self.euclidean_distance(p, p_next)

                dist.append(dist_next / dist_prev)

        return np.abs(1 - np.mean(dist))


    def flatten_coordinates(self):
        array_dims = np.shape(self.line_temp)
        yyy = np.reshape(xxx, (np.shape(xxx)[0] * np.shape(xxx)[1], 3))
        return yyy

    def save_mesh(self, file_out, alpha=0.5):
        yyy = self.flatten_coordinates()
        cloud = pv.PolyData(yyy)
        surf = cloud.delaunay_2d(alpha=alpha)

        faces = np.reshape(surf.faces, (int(len(surf.faces) / 4), 4))
        faces = faces[:, 1:]

        write_geometry(file_out, yyy, faces)










# sample data
# get lineshift
# apply lineshift
# plot sampled data on shifted line for sanity check




from nibabel.freesurfer.io import read_geometry
surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
surf_out = "/home/daniel/Schreibtisch/bb100"
ind = [40659, 189512, 181972]
vtx, fac = read_geometry(surf_in)
A = ColumnMesh(vtx, fac, ind)

xxx = A.perpendicular_line()
#xxx = A.update_coordinates(axis=0)
#xxx = A.update_coordinates(axis=1)
xxx = A.update_coordinates_proc()

yyy = A.flatten_coordinates()

A.save_mesh(surf_out)

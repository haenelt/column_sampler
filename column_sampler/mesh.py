# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.signal import find_peaks
from networkx.algorithms.shortest_paths.generic import shortest_path
from nibabel.freesurfer.io import write_geometry
from column_filter import mesh
from column_filter.io import save_overlay
from column_sampler.util import sample_line

__all__ = ["PlanarMesh", "CurvedMesh"]


class PlanarMesh(mesh.Mesh):
    LINE_LENGTH = 2
    LINE_STEP = 0.1
    NSMOOTH = 5

    # line coordinates
    x = np.arange(-LINE_LENGTH,
                  LINE_LENGTH + LINE_STEP,
                  LINE_STEP)

    def __init__(self, vtx, fac, ind, file_vol, file_deform=""):
        super().__init__(vtx, fac)
        self.ind = ind
        self.file_vol = file_vol
        self.file_deform = file_deform

    @property
    @functools.lru_cache
    def path_dijkstra(self):
        """
        https://gallantlab.github.io/pycortex/_modules/cortex/freesurfer.html
        """
        graph = nx.Graph()
        graph.add_weighted_edges_from(self._iter_edges)
        path = []
        for i in range(len(ind)-1):
            tmp = shortest_path(graph,
                                source=self.ind[i],
                                target=self.ind[i+1],
                                weight='weight',
                                method='dijkstra')
            path.extend(tmp[:-1])
        path.append(ind[-1])

        return path

    @property
    @functools.lru_cache
    def perpendicular_line(self):

        pts = []
        ind_len = len(self.path_dijkstra)
        normal = self.vertex_normals  # surface normals

        # get perpendicular line for each line vertex
        for j in range(2, ind_len - 2):

            # get local neighborhood
            nn = self.neighborhood(self.path_dijkstra[j])
            p0 = self.vtx[self.path_dijkstra[j], :]
            p = self.vtx[self.path_dijkstra] - p0

            j_next = j + self.NSMOOTH
            j_prev = j - self.NSMOOTH

            # For a stable normal line computation, neighbor points in
            # nsmooth distance are selected. This prevents the use of points
            # at both line endings. For those cases, the neighbor distance is
            # shortened.
            if j < self.NSMOOTH:
                p01 = np.mean(p[j + 1:j_next, :], axis=0)
                p02 = np.mean(p[0:j - 1, :], axis=0)
            elif j > ind_len - self.NSMOOTH:
                p01 = np.mean(p[j + 1:ind_len, :], axis=0)
                p02 = np.mean(p[j_prev:j - 1, :], axis=0)
            else:
                p01 = np.mean(p[j + 1:j_next, :], axis=0)
                p02 = np.mean(p[j_prev:j - 1, :], axis=0)

            # get smoothed surface normal
            p1 = np.mean(normal[nn], axis=0)

            # get mean vector normal to line and surface normal direction
            p21 = np.cross(p01, p1)
            p22 = np.cross(p1, p02)  # flip for consistent line direction
            p2 = (p21 + p22) / 2 + p0

            # append coordinates of perpendicular line
            pts.append(self._line_equation(self.x, p0, p2))

        return pts

    @property
    @functools.lru_cache
    def update_line(self):
        for i, bla in enumerate(self.sample_data):
            shift = self._get_shift(bla)
            self.perpendicular_line[i] = self._apply_shift(self.perpendicular_line[i], shift)

    @property
    def sample_data(self):
        data = []
        for i in range(len(self.perpendicular_line)):
            tmp = sample_line(self.perpendicular_line[i], vol_in, deform_in)
            data.append(tmp)

        return data

    @property
    def plot_data(self):
        fig, ax = plt.subplots(figsize=(5, 5))
        b = self.sample_data
        for i in range(len(b)):
            ax.plot(self.x, b[i])
        ax.set_xlabel("x in mm")
        ax.set_ylabel("fMRI contrast")
        plt.show()

    def save_overlay(self, file_out):
        ndims = np.shape(self.sample_data)
        out = np.reshape(self.sample_data, ndims[0] * ndims[1])
        save_overlay(file_out, out)

    def save_line(self, file_out):
        out = np.zeros(len(self.vtx))
        out[self.path_dijkstra] = 1
        save_overlay(file_out, out)

    def save_points(self, file_out):
        np.savez(file_out, pts=self.perpendicular_line)

    def save_mesh(self, file_out):
        second_dim = np.shape(self.perpendicular_line)[1]
        yyy = self._flatten_coordinates
        faces = []
        counter1 = 0
        length = np.shape(self.perpendicular_line)[0] * np.shape(self.perpendicular_line)[1]
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

    @staticmethod
    def euclidean_distance(pt1, pt2):
        return np.linalg.norm(pt2 - pt1)

    def _get_shift(self, data):
        xc = int(len(data) / 2)  # center coordinate
        peaks, _ = find_peaks(data)  # line peaks

        # get peak closest to center coordinate
        if len(peaks) > 1:
            peaks_diff = np.abs(peaks - xc)
            peaks = peaks[np.where(peaks_diff == np.min(peaks_diff))[0][0]]
            data_shift = peaks - xc  # distance to center coordinate
        elif len(peaks) == 1:
            peaks = peaks[0]
            data_shift = peaks - xc  # distance to center coordinate
        else:
            data_shift = []

        return data_shift

    def _apply_shift(self, coords, shift):
        xc = int(len(self.x) / 2)  # old center coordinate
        xc = xc + shift  # new center coordinate

        return self._line_equation(self.x, coords[xc, :], coords[xc + 1, :])

    @staticmethod
    def _line_equation(x, a, b):
        return np.array([a + x * (a - b) / norm(a - b) for x in x])

    @property
    def _flatten_coordinates(self):
        array_dims = np.shape(self.perpendicular_line)
        yyy = np.reshape(self.perpendicular_line, (array_dims[0] * array_dims[1], 3))
        return yyy

    @property
    def _iter_edges(self):
        for a, b, c in self.fac:
            yield a, b, self.euclidean_distance(self.vtx[a], self.vtx[b])
            yield b, c, self.euclidean_distance(self.vtx[b], self.vtx[c])
            yield a, c, self.euclidean_distance(self.vtx[a], self.vtx[c])


class CurvedMesh(PlanarMesh):
    pass


if __name__ == "__main__":
    from nibabel.freesurfer.io import read_geometry
    surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
    surf_out = "/home/daniel/Schreibtisch/bb100"
    ind = [106481, 103769, 101279, 98771]
    v, f = read_geometry(surf_in)
    vol_in = "/home/daniel/Schreibtisch/data/data_sampler/vol/Z_all_left_right_GE_EPI3.nii"
    deform_in = "/home/daniel/Schreibtisch/data/data_sampler/vol/source2target.nii.gz"

    A = PlanarMesh(v, f, ind, vol_in, deform_in)
    A.save_mesh(surf_out)

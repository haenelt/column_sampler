# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from networkx.algorithms.shortest_paths.generic import shortest_path
from column_filter import mesh
from column_filter.io import save_overlay
from column_sampler.util import sample_data

__all__ = ["PlanarMesh", "CurvedMesh"]


class PlanarMesh(mesh.Mesh):
    ## noinspection PyUnresolvedReferences
    """Mesh with planar shape.

    Bla...

    Parameters
    ----------
    vtx : (N,3) np.ndarray
        Base directory of subject folders.
    fac : (M,3) np.ndarray
        sdf.
    idx : (O) list
        sdf.

    Attributes
    ----------
    LINE_LENGTH : int
        bla.
    LINE_STEP : float
        must be chosen so that the coordinates length is odd.

    Raises
    ------
    FileNotFoundError :
        If `dir_base` is not found or if not all subfolders in the subject
        folder are found.
    ValueError :
        If `sub` deviates from the names on my google drive.

    """

    LINE_LENGTH = 2
    LINE_STEP = 0.1
    NSMOOTH = 5

    # line coordinates
    x = np.arange(-LINE_LENGTH,
                  LINE_LENGTH + LINE_STEP,
                  LINE_STEP)

    def __init__(self, vtx, fac, idx):
        super().__init__(vtx, fac)
        self.idx = idx

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
                                source=self.idx[i],
                                target=self.idx[i + 1],
                                weight='weight',
                                method='dijkstra')
            path.extend(tmp[:-1])
        path.append(ind[-1])

        return path

    @property
    @functools.lru_cache
    def line_coordinates(self):
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

    def shift_coordinates(self, file_vol, file_deform):
        data = sample_data(self.line_coordinates, file_vol, file_deform)
        self._plot_data(file_vol, file_deform)
        for i, bla in enumerate(data):
            shift = self._get_shift(bla)
            xc = int(len(self.x) / 2)  # old center coordinate
            xc = xc + shift  # new center coordinate
            pt1 = self.line_coordinates[i][xc, :]
            pt2 = self.line_coordinates[i][xc + 1, :]
            self.line_coordinates[i] = self._line_equation(self.x, pt1, pt2)
        self._plot_data(file_vol, file_deform)

        return self.line_coordinates

    def save_line(self, file_out):
        out = np.zeros(len(self.vtx))
        out[self.path_dijkstra] = 1
        save_overlay(file_out, out)

    def _plot_data(self, file_vol, file_deform):
        fig, ax = plt.subplots(figsize=(5, 5))
        b = sample_data(self.line_coordinates, file_vol, file_deform)
        for i in range(len(b)):
            ax.plot(self.x, b[i])
        ax.set_xlabel("x in mm")
        ax.set_ylabel("fMRI contrast")
        plt.show()

    @property
    def _iter_edges(self):
        for a, b, c in self.fac:
            yield a, b, self._euclidean_distance(self.vtx[a], self.vtx[b])
            yield b, c, self._euclidean_distance(self.vtx[b], self.vtx[c])
            yield a, c, self._euclidean_distance(self.vtx[a], self.vtx[c])

    @staticmethod
    def _get_shift(data):
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
            data_shift = 0

        return data_shift

    @staticmethod
    def _euclidean_distance(pt1, pt2):
        return np.linalg.norm(pt2 - pt1)

    @staticmethod
    def _line_equation(x, a, b):
        return np.array([a + i * (b - a) / np.linalg.norm(a - b) for i in x])

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, i):
        # only checks for normal multi-dimensional lists (array)
        if not isinstance(i, list) or isinstance(i[0], list):
            raise ValueError("Index list is not a one-dimensional list!")

        self._idx = i


class CurvedMesh(PlanarMesh):
    MAX_ITER = 100000
    COST_THRES = 1e-4
    DIST_RATIO = 0.1
    REPOSITION_STEP = 10000
    CHECK_STEP = 1000

    def __init__(self, vtx, fac, idx):
        super().__init__(vtx, fac, idx)

    def project_coordinates_sequence(self, axis=(0, 1)):
        for i in axis:
            print(i)
            self.project_coordinates(i)

        return self.line_coordinates

    def project_coordinates(self, axis=0):
        # pts -> lines x pts x coords
        pts = self.line_coordinates  # type: list
        counter = 0
        while counter < self.MAX_ITER:
            counter += 1
            pt_fix = int(len(self.line_coordinates[0])/2)

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

            if y_random == pt_fix:
                continue

            dist_prev = self._euclidean_distance(p, p_prev)
            dist_next = self._euclidean_distance(p, p_next)
            if dist_next / dist_prev > self.DIST_RATIO:

                # new line point coordinates is the mean of neighboring
                # coordinates
                pts[x_random][y_random][0] = (p_prev[0] + p_next[0]) / 2
                pts[x_random][y_random][1] = (p_prev[1] + p_next[1]) / 2
                pts[x_random][y_random][2] = (p_prev[2] + p_next[2]) / 2

            # reposition all line coordinates
            if not np.mod(counter, self.REPOSITION_STEP):
                for i, line in enumerate(pts):
                    pts[i] = self._reposition_mesh(line)

            if not np.mod(counter, self.CHECK_STEP):
                cost = self._check_homogeneity(pts, axis=axis)
                if cost < self.COST_THRES:
                    for i, line in enumerate(pts):
                        pts[i] = self._reposition_mesh(line)
                    break

        return pts

    def _reposition_mesh(self, pts):
        # reposition in the sense of vertex shifting
        # get closest vertex
        # get normal
        # for each y -> reposition with formula
        res = []
        for bla in pts:
            ind_here = self._closest_point(bla)
            n = self.vertex_normals[ind_here, :]
            res.append(bla-np.dot(np.outer(n, n), bla-self.vtx[ind_here, :]))

        return res

    def _closest_point(self, pt):
        vtx_tmp = self.vtx - pt
        dist = np.sqrt(vtx_tmp[:, 0]**2+vtx_tmp[:, 1]**2+vtx_tmp[:, 2]**2)
        idc = np.where(dist == np.min(dist))[0][0]

        return idc

    def _check_homogeneity(self, arr, axis=0):
        dist = []
        for i in range(1, np.shape(arr)[0]-1):
            for j in range(1, np.shape(arr)[1]-1):

                p = arr[i][j]

                # random line point
                if axis == 0:
                    p_prev = arr[i - 1][j]
                    p_next = arr[i + 1][j]
                elif axis == 1:
                    p_prev = arr[i][j - 1]
                    p_next = arr[i][j + 1]
                else:
                    raise ValueError("Invalid argument for axis!")

                dist_prev = self._euclidean_distance(p, p_prev)
                dist_next = self._euclidean_distance(p, p_next)

                dist.append(dist_next / dist_prev)

        return np.abs(1 - np.mean(dist))


if __name__ == "__main__":
    import os
    from nibabel.freesurfer.io import read_geometry
    from column_sampler.io import save_coords
    surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
    surf_out = "/home/daniel/Schreibtisch/bb100.npz"
    ind = [106481, 103769, 101279, 98771]
    v, f = read_geometry(surf_in)
    dir_base = "/home/daniel/Schreibtisch/data/data_sampler/vol"
    vol_in = os.path.join(dir_base, "Z_all_left_right_GE_EPI3.nii")
    deform_in = os.path.join(dir_base, "source2target.nii.gz")

    A = CurvedMesh(v, f, ind)
    coords = A.project_coordinates_sequence()
    save_coords(surf_out, coords)

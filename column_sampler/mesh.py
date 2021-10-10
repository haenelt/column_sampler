# -*- coding: utf-8 -*-
"""Triangle surface mesh definition."""

import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from networkx.algorithms.shortest_paths.generic import shortest_path
from column_filter import mesh
from column_filter.io import save_overlay
from util import sample_data, flatten_array, unflatten_array

__all__ = ["PlanarMesh", "CurvedMesh"]


class PlanarMesh(mesh.Mesh):
    """Mesh with planar shape.

    Defines a planar mesh based on a list of vertex indices. Vertex indices from
    the list are connected to a line by their shortest path. Lines orthogonal to
    the resulting line are then computed which are centered on the vertex line.
    Optionally, line centers can be shifted based on a separate contrast file.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N, 3)
        Vertex array.
    fac : np.ndarray, shape=(M, 3)
        Corresponding face array.
    idx : list
        List of vertex indices.

    Attributes
    ----------
    LINE_LENGTH : int
        Line length of orthogonal lines from the middle point, i.e., the whole
        line is twice as long.
    LINE_STEP : float
        Step size between neighboring line coordinates. The step size must be
        chosen so that the line array length is odd.
    NSMOOTH : int
        Number of neighboring line indices used for computation of line normals.
    x : np.ndarray, shape=(N,)
        Resulting line coordinates.

    Raises
    ------
    ValueError :
        If `vtx` has an invalid shape.
    ValueError :
        If `fac` has an invalid shape or does not match the vertex array `vtx`.
    ValueError :
        If `idx` is not a one-dimensional list.

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
        """Returns a list of vertex indices containing the path which connects
        input vertex indices using the Dijkstra algorithm. The code is based on
        [1]_.

        Returns
        -------
        path : list
            Vertex indices forming a path on the input surface mesh.

        References
        ----------
        .. [1] https://gallantlab.github.io/pycortex/_modules/cortex/freesurfer.html

        """

        graph = nx.Graph()
        graph.add_weighted_edges_from(self._iter_edges)
        path = []
        for i in range(len(self.idx)-1):
            tmp = shortest_path(graph,
                                source=self.idx[i],
                                target=self.idx[i + 1],
                                weight='weight',
                                method='dijkstra')
            path.extend(tmp[:-1])
        path.append(self.idx[-1])

        return path

    @property
    @functools.lru_cache
    def line_coordinates(self):
        """Computes line coordinates perpendicular to a line on a surface mesh.
        The resulting array consists of N lines with M 3D coordinates.

        Returns
        -------
        np.ndarray, shape=(N, M, 3)
            Coordinate array of lines.

        """

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

        return np.array(pts)

    def shift_coordinates(self, file_vol, file_deform=None):
        """Shifts line coordinates by centering midpoints to a reference image.
        Distributions of sampled data along lines are plotted before and after
        shifting for sanity check.

        Parameters
        ----------
        file_vol : str
            Reference image (nifti volume).
        file_deform : str, optional
            Deformation field (4D nifti volume).

        Returns
        -------
        np.ndarray, shape=(N, M, 3)
            Shifted coordinate array of lines.


        """

        data = self._sample_data(file_vol, file_deform)
        self._plot_data(file_vol, file_deform)
        for i, y in enumerate(data):
            shift = self._get_shift(y)
            xc = int(len(self.x) / 2)  # old center coordinate
            xc = xc + shift  # new center coordinate
            pt1 = self.line_coordinates[i][xc, :]
            pt2 = self.line_coordinates[i][xc + 1, :]
            self.line_coordinates[i] = self._line_equation(self.x, pt1, pt2)
        self._plot_data(file_vol, file_deform)

        return self.line_coordinates

    def save_line(self, file_out):
        """Saves Dijkstra's shortest path as MGH overlay.

        Parameters
        ----------
        file_out : str
            File name of output file.

        Returns
        -------
        None.

        """

        out = np.zeros(len(self.vtx))
        out[self.path_dijkstra] = 1
        save_overlay(file_out, out)

    def _plot_data(self, file_vol, file_deform=None):
        """Plot distribution of sampled data.

        Parameters
        ----------
        file_vol : str
            Reference image (nifti volume).
        file_deform : str, optional
            Deformation field (4D nifti volume).

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(5, 5))
        data = self._sample_data(file_vol, file_deform)
        for y in data:
            ax.plot(self.x, y)
        ax.set_xlabel("x in mm")
        ax.set_ylabel("fMRI contrast")
        plt.show()

    def _sample_data(self, file_vol, file_deform=None):
        """Sample data from reference volume.

        Parameters
        ----------
        file_vol : str
            Reference image (nifti volume).
        file_deform : str, optional
            Deformation field (4D nifti volume).

        Returns
        -------
        np.ndarray, shape=(N, M)
            Unflattened data array.

        """

        coords_flat = flatten_array(self.line_coordinates)
        data = sample_data(coords_flat, file_vol, file_deform)
        return unflatten_array(data, self.line_coordinates)

    @property
    def _iter_edges(self):
        """Creates iterators for mesh edges with associated Euclidean distances.

        Returns
        -------
        None.

        """

        for a, b, c in self.fac:
            yield a, b, self._euclidean_distance(self.vtx[a], self.vtx[b])
            yield b, c, self._euclidean_distance(self.vtx[b], self.vtx[c])
            yield a, c, self._euclidean_distance(self.vtx[a], self.vtx[c])

    @staticmethod
    def _get_shift(data):
        """Finds the peak of a data array and returns the distance to its center
        position. The center position is the midpoint of the array.

        Parameters
        ----------
        data : np.ndarray, shape=(N,)
            One-dimensional data array.

        Returns
        -------
        data_shift : float
            Distance to center position.

        """

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
        """Computes Euclidean distance between two points.

        Parameters
        ----------
        pt1 : np.ndarray, shape=(3,)
            Coordinates of first point.
        pt2 : np.ndarray, shape=(3,)
            Coordinates of second  point.

        Returns
        -------
        float
            Euclidean distance.

        """

        return np.linalg.norm(pt2 - pt1)

    @staticmethod
    def _line_equation(x, a, b):
        """Computes line coordinates of a line defined by two points for a list
        of line positions.

        Parameters
        ----------
        x : list
            One-dimensional list of line positions for which coordinates are
            computed.
        a : np.ndarray, shape=(3,)
            Coordinates of first line point.
        b : np.ndarray, shape=(3,)
            Coordinates of second line point.

        Returns
        -------
        np.ndarray, shape=(N, 3)
            Line coordinates.

        """

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
    """Mesh with curved shape.

    Defines a curved mesh based on a list of vertex indices. Vertex indices from
    the list are connected to a line by their shortest path. Lines orthogonal to
    the resulting line are then computed which are centered on the vertex line.
    Optionally, line centers can be shifted based on a separate contrast file.
    Orthogonal lines are then repositioned to lie on the the surface mesh.

    Parameters
    ----------
    vtx : np.ndarray, shape=(N, 3)
        Vertex array.
    fac : np.ndarray, shape=(M, 3)
        Corresponding face array.
    idx : list
        List of vertex indices.

    Attributes
    ----------
    MAX_ITER : int
        Maximal number of iterations for line repositioning.
    COST_THRES : float
        Break condition for line repositioning.
    DIST_RATIO : float
        Target distance ratio between neighboring points.
    REPOSITION_STEP : int
        Number of iterations between line repositioning.
    CHECK_STEP : int
        Number of iterations between homogeneity checks.

    Raises
    ------
    ValueError :
        If `vtx` has an invalid shape.
    ValueError :
        If `fac` has an invalid shape or does not match the vertex array `vtx`.
    ValueError :
        If `idx` is not a one-dimensional list.

    """

    MAX_ITER = 100000
    COST_THRES = 1e-4
    DIST_RATIO = 0.1
    REPOSITION_STEP = 10000
    CHECK_STEP = 1000

    def __init__(self, vtx, fac, idx):
        super().__init__(vtx, fac, idx)

    def project_coordinates_sequence(self, axis=(0, 1)):
        """Runs mesh repositioning multiple times along a sequence of axes along
        which mesh repositioning is performed.

        Parameters
        ----------
        axis : tuple
            Sequence of axes.

        Returns
        -------
        np.ndarray, shape=(N, M, 3)
            Coordinate array of curved lines.

        """

        for i in axis:
            print("Run projection for axis "+str(i))
            self.project_coordinates(i)

        return self.line_coordinates

    def project_coordinates(self, axis=0):
        """Runs mesh repositioning along a defined axis.

        Parameters
        ----------
        axis : int, optional
            Axis along repositioning is performed. Valid axis parameters are 0
            or 1.

        Returns
        -------
        np.ndarray, shape=(N, M, 3)
            Coordinate array of curved lines.

        """

        pts = self.line_coordinates
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

            # check break condition
            if not np.mod(counter, self.CHECK_STEP):
                cost = self._check_homogeneity(pts, axis=axis)
                if cost < self.COST_THRES:
                    for i, line in enumerate(pts):
                        pts[i] = self._reposition_mesh(line)
                    break

        return np.array(pts)

    def _reposition_mesh(self, pts):
        """Performs remeshing in the sense of vertex shifting. For a coordinate,
        the closest vertex of the surface mesh is found and the coordinate is
        then projected along the vertex normal onto the surface mesh.

        Parameters
        ----------
        pts : np.ndarray, shape=(N, 3)
            List of coordinates.

        Returns
        -------
        np.ndarray, shape=(N, 3)
            Repositioned list of coordinates.

        """

        pts_proj = []
        for i in pts:
            id0 = self._closest_point(i)
            n = self.vertex_normals[id0, :]
            pts_proj.append(i-np.dot(np.outer(n, n), i-self.vtx[id0, :]))

        return np.array(pts_proj)

    def _closest_point(self, pt):
        """Finds vertex which has smallest Euclidean distance to a given point.

        Parameters
        ----------
        pt : np.ndarray, shape=(3,)
            3D coordinate.

        Returns
        -------
        int
            Index of closest vertex.

        """

        v = self.vtx - pt
        dist = np.sqrt(v[:, 0]**2+v[:, 1]**2+v[:, 2]**2)

        return np.where(dist == np.min(dist))[0][0]

    def _check_homogeneity(self, pts, axis=0):
        """Checks the homogeneity of a coordinate array by computing the mean
        ratio of distances to both neighbor points along one axis. A homogeneous
        array should return a value close to 0.

        Parameters
        ----------
        pts : np.ndarray, shape=(N, 3)
            List of coordinates.
        axis : int, optional
            Axis along repositioning is performed. Valid axis parameters are 0
            or 1.

        Returns
        -------
        float
            Distance ratio.

        """

        dist = []
        for i in range(1, np.shape(pts)[0] - 1):
            for j in range(1, np.shape(pts)[1] - 1):

                p = pts[i][j]
                if axis == 0:
                    p_prev = pts[i - 1][j]
                    p_next = pts[i + 1][j]
                elif axis == 1:
                    p_prev = pts[i][j - 1]
                    p_next = pts[i][j + 1]
                else:
                    raise ValueError("Invalid argument for axis!")

                dist_prev = self._euclidean_distance(p, p_prev)
                dist_next = self._euclidean_distance(p, p_next)
                dist.append(dist_next / dist_prev)

        return np.abs(1 - np.mean(dist))

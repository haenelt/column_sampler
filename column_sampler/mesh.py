# -*- coding: utf-8 -*-
"""Utility functions for triangle surface mesh."""

import functools
import numpy as np
import networkx as nx
from column_filter import mesh
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


from nibabel.freesurfer.io import read_geometry

surf_in = "/home/daniel/Schreibtisch/data/data_sampler/surf/lh.layer_5"
ind = [159488, 30645, 171581]

vtx, fac = read_geometry(surf_in)
A = Mesh(vtx, fac)
res = A.path_dijkstra(ind)

for i in range(len(res)-1):
    if res[i] == res[i+1]:
        print("True")
    else:
        print("False")

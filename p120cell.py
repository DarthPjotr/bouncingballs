# pylint: disable=C,I
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable

# thanks to data from:
# http://www.meshcompression.org/random-stuff/13-vertices-and-cells-of-a-120-cell-4d-polychoron

import math
import networkx as nx
import matplotlib.pyplot as plt

from pprint import pprint as pp  # pylint: disable=unused-import
from p120cell_def import p120cell_edges

def _get_neighbours():
    path = "d:\\temp\\120cell\\neighbours.txt"
    with open(path, encoding="utf8") as file:
        lines = file.readlines()

    neighbours = []
    for line in lines:
        index, rest = line.split(":")
        neighbours_ = rest.split(",")
        index = int(index)
        neighbours_ = [int(n) for n in neighbours_]
        neighbours_.insert(0, index)
        neighbours.append(neighbours_)

    return neighbours

def _get_indices():
    indices = [0,1,2,3,4,5,6,7]

    v0 = 0
    v1 = 1
    v2 = math.sqrt(5)
    v3 = (1+v2)/2
    v4 = 1/(v3*v3)
    v5 = v3*v3
    v6 = 1/v3
    v7 = 2

    indices[0] = v0
    indices[1] = v1
    indices[2] = v2
    indices[3] = v3
    indices[4] = v4
    indices[5] = v5
    indices[6] = v6
    indices[7] = v7

    return indices

def _get_vertices(indices):
    path = "d:\\temp\\120cell\\vertices.txt"
    with open(path, encoding="utf8") as file:
        lines = file.readlines()

    vertices = []
    for line in lines:
        i, rest = line.split(":")
        indices_ = rest.split(",")
        i = int(i)
        indices_ = [int(n) for n in indices_]
        coords = []
        for ind in indices_:
            coord = indices[abs(ind)]
            if ind < 0:
                coord = -coord

            coords.append(coord)
        vertices.insert(i, tuple(coords))

    return vertices

def _load_120cell():
    edges = []
    indices = _get_indices()
    vertices = _get_vertices(indices)
    print(len(vertices))
    G = nx.Graph()

    neighbours = _get_neighbours()
    print(len(neighbours))

    for n in neighbours:
        i = n[0]
        v1 = vertices[i]
        rest = n[1:]
        for r in rest:
            v2 = vertices[r]
            edge = (v1, v2)
            edges.append(edge)
            G.add_edge(v1, v2)

    path = "d:\\temp\\p120cell_def.py"
    with open(path, encoding="utf8", mode="w") as file:
        pp(object=edges, stream=file)
    return G


def create_120cell():
    G = nx.Graph()

    for (v1, v2) in p120cell_edges:
        G.add_edge(v1, v2)

    return G


def main():

    # The 120-cell with long radius √8 = 2√2 ≈ 2.828 has edge length 2/φ2 = 3−√5 ≈ 0.764.

    G = create_120cell()
    # colors = nx.greedy_color(G)
    max_degree = max(degree for (node, degree) in G.degree())
    print(max_degree)
    colors = nx.equitable_color(G, max_degree+1)
    print(set(colors.values()))

    pp(G)
    # print(colors)
    nx.draw(G)
    # for i, node in enumerate(G.nodes):
    #     print(i, node)

    # for i, edge in enumerate(G.edges):
    #     print(i, edge)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()




if __name__ == "__main__":
    main()

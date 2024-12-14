import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


def _desc_sort(d):
    keys = list(d.keys())
    values = list(d.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_d = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_d

def deg_centrality(graph):
    cent = nx.degree_centrality(graph)
    sorted_cent = _desc_sort(cent)

    return sorted_cent

def closeness_centrality(graph):
    cent = nx.closeness_centrality(graph)
    sorted_cent = _desc_sort(cent)

    return sorted_cent

def betw_centrality(graph):
    cent = nx.betweenness_centrality(graph)
    sorted_cent = _desc_sort(cent)

    return sorted_cent

def eigen_centrality(graph):
    cent = nx.eigenvector_centrality(graph, max_iter=500)
    sorted_cent = _desc_sort(cent)

    return sorted_cent

def get_idx(eigenvalues):
    diff = 1e-4

    for i in range(len(eigenvalues) - 1):
        curr = eigenvalues[i]
        ne = eigenvalues[i+1]

        if np.abs(curr - ne) > diff and ne > 1e-6:
            return i+1, i+2

    return None

def topk_spectra(graph, k):
    A = nx.adjacency_matrix(graph)
    L = laplacian(A, normed=True)

    eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
    yi, yj = get_idx(eigenvalues)

    vectors = eigenvectors[:, np.argsort(eigenvalues)[yi:yi+10]]
    node_importance = {node: vectors[i] for i, node in enumerate(graph.nodes)}

    return node_importance

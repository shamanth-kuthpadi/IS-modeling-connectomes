import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import community as community_louvain 
import numpy as np
import math

# sorter for dictionary
def _desc_sort(d):
    keys = list(d.keys())
    values = list(d.values())
    sorted_value_index = np.argsort(values)[::-1]
    sorted_d = {keys[i]: values[i] for i in sorted_value_index}
    return sorted_d

# Centrality metrics

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

def get_all_centrality_metrics(graph):
    store = {}
    
    deg_cent = deg_centrality(graph)
    close_cent = closeness_centrality(graph)
    betw_cent = betw_centrality(graph)
    eig_cent = eigen_centrality(graph)

    store['degree_centrality'] = deg_cent
    store['closeness_centrality'] = close_cent
    store['betweeness_centrality'] = betw_cent
    store['eigenvector_centrality'] = eig_cent

    print(f"Node with highest degree centrality: {list(deg_cent)[0]} = {deg_cent[list(deg_cent)[0]]}")
    print(f"Node with highest closeness centrality: {list(close_cent)[0]} = {close_cent[list(close_cent)[0]]}")
    print(f"Node with highest betweeness centrality: {list(betw_cent)[0]} = {betw_cent[list(betw_cent)[0]]}")
    print(f"Node with highest eigenvector centrality: {list(eig_cent)[0]} = {eig_cent[list(eig_cent)[0]]}")

    return store


def get_nodes_by_cluster(partition):
    """
    Organize nodes by their cluster from the partition dictionary.
    
    Params:
        partition: dict where keys are nodes and values are cluster IDs
    
    Returns:
        clusters: dict where keys are cluster IDs and values are lists of nodes in each cluster
    """
    clusters = {}
    
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    return clusters

def print_node_attributes_by_cluster(graph, partition):
    """
    Print the attributes of nodes grouped by their cluster.
    
    Params:
        graph: NetworkX graph
        partition: dict where keys are node IDs and values are cluster IDs
    
    Returns:
        None (prints the nodes' attributes for each cluster)
    """
    clusters = get_nodes_by_cluster(partition)

    for cluster_id, nodes in clusters.items():
        print(f"Cluster {cluster_id}:")
        for node in nodes:
            attributes = graph.nodes[node]
            print(f"Node: {node}, Attributes: {attributes}")
        print("\n" + "="*50 + "\n")

def print_edge_attributes_by_cluster(graph, partition):
    """
    Print the attributes of edges grouped by their cluster.
    
    Params:
        graph: NetworkX graph
        partition: dict where keys are node IDs and values are cluster IDs
    
    Returns:
        None (prints the edges' attributes for each cluster)
    """
    # Group nodes by cluster ID
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # Iterate over clusters to find edges within each cluster
    for cluster_id, nodes in clusters.items():
        print(f"Cluster {cluster_id}:")
        mean_fa = 0
        n = 0
        # Filter edges where both nodes are in the same cluster
        for u, v in graph.edges(nodes):
            if partition[u] == partition[v]:  # Ensure both nodes are in the same cluster
                attributes = graph.edges[u, v]
                mean_fa += attributes['FA_mean']
                n += 1
                print(f"Edge: ({u}, {v}), Attributes: {attributes}")
        
        print("\n" + "="*50 + "\n")
    
        print(f"Mean FA = {mean_fa / n}")

def get_edge_weight_dist(graph, partition, unique_clusters, subplots=False):
    if subplots is False:    
        plt.figure(figsize=(12, 8))
        for cluster in unique_clusters:
            cluster_edges = [
                data['FA_mean'] for u, v, data in graph.edges(data=True)
                if partition[u] == cluster and partition[v] == cluster
            ]

            plt.hist(cluster_edges, bins=30, alpha=0.5, label=f'Cluster {cluster}', edgecolor='black')
        plt.title('Edge Weight Distribution by Cluster (FA_mean)')
        plt.xlabel('FA_mean')
        plt.ylabel('Frequency')
        plt.legend(title="Clusters")
        plt.show()
    else:
        num_clusters = len(unique_clusters)
        cols = math.ceil(math.sqrt(num_clusters))
        rows = math.ceil(num_clusters / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows), squeeze=False)

        for idx, cluster in enumerate(unique_clusters):
            row = idx // cols
            col = idx % cols
            cluster_edges = [
                data['FA_mean'] for u, v, data in graph.edges(data=True)
                if partition[u] == cluster and partition[v] == cluster
            ]
            axs[row, col].hist(cluster_edges, bins=30, alpha=0.75, edgecolor='black')
            axs[row, col].set_title(f'Cluster {cluster}')
            axs[row, col].set_xlabel('FA_mean')
            axs[row, col].set_ylabel('Frequency')

        for i in range(num_clusters, rows * cols):
            fig.delaxes(axs[i // cols, i % cols])

        plt.suptitle('Edge Weight Distribution by Cluster (FA_mean)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def fget_clusters(partition):
    clusters = get_nodes_by_cluster(partition)

    for cluster_id, nodes in clusters.items():
        print(f"Cluster {cluster_id}:")
        print(nodes)
        print(f"Number of nodes: {len(nodes)}")
        print()
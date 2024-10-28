import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import community as community_louvain 
import numpy as np
import math

def visualize_connectome(file, sub_id, cmap='YlGnBu'):
    '''
    Params:
        file: file path for a .graphml extension
        sub_id: id of the subject
        cmap: color map to apply to the visualization
    Returns:
        Networkx object
    '''

    graph = nx.read_graphml(file)
    A = nx.adjacency_matrix(graph)

    plt.imshow(A.toarray(), cmap=cmap)
    plt.colorbar()
    plt.title(f'{sub_id} Connectome Matrix')
    plt.show()

    return graph

def visualize_network(file, sub_id, cmap=plt.cm.plasma, node_color='skyblue', highlight=[], clustering=False, use_3d=False, plot=True):
    '''
    Params:
        file: file path for a .graphml extension
        sub_id: id of the subject
        cmap: color map to apply to the visualization
        node_color: color of the nodes
    Returns:
        Networkx object

    Ensure that the formatting of the graphml has attributes dedicated for coordinate positions
    and also for the FA mean pertaining to edge weights between nodes
    '''

    graph = nx.read_graphml(file)

    # removing nodes that do not attribute to connectivity
    isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
    graph.remove_nodes_from(isolated_nodes)

    # utilizing the x and y positions of each node in the connectome
    positions = {}
    for node, data in graph.nodes(data=True):
        if 'dn_position_x' in data and 'dn_position_y' in data:
            positions[node] = (data['dn_position_x'], data['dn_position_y'])
            if 'dn_position_z' in data and use_3d is True:
                positions[node] = (data['dn_position_x'], data['dn_position_y'], data['dn_position_z'])

    # utilizing the FA mean of edges between nodes as an indicator of edge strengths
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        fa_mean = data['FA_mean']
        edge_colors.append(fa_mean)

    # normalizing the FA means -- seems to be common practice when applying a color map
    edge_colors = np.array(edge_colors)
    
    norm = plt.Normalize(vmin=edge_colors.min(), vmax=edge_colors.max())

    # if we are using 3d visualization then we should set up a projection
    if use_3d:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
    else:
        plt.figure(figsize=(12, 12))

    # if we want to see the clusters using the Louvain algorithm then we should do the following
    if clustering is True:
        partition = community_louvain.best_partition(graph)
        unique_clusters = set(partition.values())
        cluster_colors = cm.get_cmap('Spectral', len(unique_clusters))

        n_colors = [cluster_colors(partition[node]) if node in partition else node_color for node in graph.nodes]
    else:
        n_colors = ['red' if node_name in highlight else node_color for node_name in list(graph.nodes)]

    if use_3d:
        x_vals, y_vals, z_vals = zip(*positions.values())
        ax.scatter(x_vals, y_vals, z_vals, c=n_colors, s=30, edgecolors='none', alpha=0.9)

        for u, v, data in graph.edges(data=True):
            x_vals = [positions[u][0], positions[v][0]]
            y_vals = [positions[u][1], positions[v][1]]
            z_vals = [positions[u][2], positions[v][2]]
            ax.plot(x_vals, y_vals, z_vals, color=cmap(norm(data['FA_mean'])), alpha=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    else:
        nx.draw_networkx_nodes(graph, pos=positions, node_size=10, node_color=n_colors)
        if clustering is True:
            nx.draw_networkx_edges(graph, pos=positions, width=0.5, edge_color=edge_colors, edge_cmap=cmap, alpha=0.1)
        else:
            nx.draw_networkx_edges(graph, pos=positions, width=1, edge_color=edge_colors, edge_cmap=cmap, alpha=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if use_3d:
        fig.colorbar(sm, ax=ax, label="FA_mean")
    else:
        plt.colorbar(sm, ax=plt.gca(), label="FA_mean")

    plt.title(f'{sub_id} Connectome Network')
    if plot is True:
        plt.show()

    return graph, partition, unique_clusters


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
    cent = nx.eigenvector_centrality(graph)
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
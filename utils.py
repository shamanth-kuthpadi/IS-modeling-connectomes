import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Visualization definitions

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

def visualize_network(file, sub_id, cmap=plt.cm.plasma, node_color='skyblue', highlight=[]):
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

    # utilizing the FA mean of edges between nodes as an indicator of edge strengths
    edge_colors = []
    for u, v, data in graph.edges(data=True):
        fa_mean = data['FA_mean']
        edge_colors.append(fa_mean)

    # normalizing the FA means -- seems to be common practice when applying a color map
    edge_colors = np.array(edge_colors)
    
    cmap = cmap
    norm = plt.Normalize(vmin=edge_colors.min(), vmax=edge_colors.max())

    plt.figure(figsize=(12, 12))


    n_colors = ['red' if node_name in highlight else node_color for node_name in list(graph.nodes)]

    nx.draw_networkx_nodes(graph, pos=positions, node_size=10, node_color=n_colors)

    nx.draw_networkx_edges(graph, pos=positions, width=1, edge_color=edge_colors, edge_cmap=cmap, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.colorbar(sm, ax=plt.gca(), label="FA_mean")

    plt.title(f'{sub_id} Connectome Network')
    plt.show()

    return graph


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

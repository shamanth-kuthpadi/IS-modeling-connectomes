import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import community as community_louvain 
import numpy as np
import math

class Connectome:
    
    def __init__(self, file, label):
        graph = nx.read_graphml(file)
        self.label = label
        self.graph = graph
        self.matrix = None
        self.net = None
    
    def read_matrix(self):
        A = nx.adjacency_matrix(self.graph)
        self.matrix = A
        return A
    
    def plot_matrix(self, cmap='YlGnBu'):
        plt.imshow(self.matrix.toarray(), cmap=cmap)
        plt.colorbar()
        plt.title(f'{self.label} Connectome Matrix')
        plt.show()

    def read_net(self, use_3d=False):
        graph = self.graph
        isolated_nodes = [node for node, degree in graph.degree() if degree == 0]
        graph.remove_nodes_from(isolated_nodes)

        positions = {}
        for node, data in graph.nodes(data=True):
            if 'dn_position_x' in data and 'dn_position_y' in data:
                positions[node] = (data['dn_position_x'], data['dn_position_y'])
                if 'dn_position_z' in data and use_3d is True:
                    positions[node] = (data['dn_position_x'], data['dn_position_y'], data['dn_position_z'])

        edge_colors = []
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            edge_count += 1
            fa_mean = data['FA_mean']
            edge_colors.append(fa_mean)
        edge_colors = np.array(edge_colors)

        self.net = (graph, positions, edge_colors)
    
    def perform_clustering(self, graph):
        partition = community_louvain.best_partition(graph)
        unique_clusters = set(partition.values())
        cluster_colors = cm.get_cmap('Spectral', len(unique_clusters))

        return partition, unique_clusters, cluster_colors

    def plot_net(self, cmap=plt.cm.plasma, node_color='skyblue', highlight=[], use_3d=False, clustering=False):
        graph, positions, edge_colors = self.net

        norm = plt.Normalize(vmin=edge_colors.min(), vmax=edge_colors.max())

        if use_3d:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
        else:
            plt.figure(figsize=(12, 12))

        if clustering is True:
            partition, unique_clusters, cluster_colors = self.perform_clustering(graph)
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

        plt.title(f'{self.label} Connectome Network')
        plt.show()
    

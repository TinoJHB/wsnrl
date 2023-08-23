import random
import numpy as np
from node import Node
from kmeansjeudi import ElbowMethodKMeans

class NetworkTopology:
    def __init__(self, num_nodes, comm_range, int_range, max_iterations, E_level, loc, RSSI, m):
        self.num_nodes = num_nodes
        self.comm_range = comm_range
        self.int_range = int_range
        self.max_iterations = max_iterations
        self.E_level = E_level
        self.loc = loc
        self.RSSI = RSSI
        self.m = m
        # Randomly generate N nodes within given range of 300mx300m or 1000mx1000m
        self.nodes = self.generate_random_nodes(num_nodes)

        # Determine correlation and interference relationships between nodes
        self.correlations = {}
        self.interferences = {}

        for i in range(num_nodes):
            corr_set = set()
            int_set = set()

            for j in range(i + 1, num_nodes):
                dist_ij_sqrd = (self.nodes[i].loc[0] - self.nodes[j].loc[0]) ** 2 + (self.nodes[i].loc[1] - self.nodes[j].loc[1]) ** 2

                if dist_ij_sqrd <= comm_range ** 2.:
                    corr_set.add(j)

                    if dist_ij_sqrd <= int_range ** 2:
                        int_set.add(j)

                elif dist_ij_sqrd <= int_range ** 2:
                    int_set.add(j)

            if len(corr_set) > 0:
                self.correlations[i] = corr_set

            if len(int_set) > 0:
                self.interferences[i] = int_set

    def generate_random_nodes(self, num_nodes):
        nodes = []
        for _ in range(num_nodes):
            E_level = np.random.rand() * 100
            loc = np.array([random.uniform(0, 1), random.uniform(0, 1)])
            RSSI = np.random.rand() * (-100)
            node = Node(E_level, loc, RSSI)
            nodes.append(node)

        return nodes

    def form_clusters(self, K):
        """Forms K clusters using K-Means algorithm."""
        # Extract numerical features from Node objects
        node_array = np.array([[node.E_level, node.loc[0], node.loc[1], node.RSSI] for node in self.nodes])
        print(node_array.dtype)

        # Perform K-Means clustering with the Elbow Method
        kmeans = ElbowMethodKMeans(K, self.max_iterations, final_model=None)
        kmeans_labels = kmeans.fit(node_array)

        # Create a list to store the nodes in each cluster
        cluster_list = [[] for _ in range(K)]

        for l, node_id in enumerate(range(len(kmeans_labels))):
            cluster_list[kmeans_labels[l]].append(node_id)

        # Select a cluster head for each cluster
        cluster_heads = []

        for cluster_nodes in cluster_list:
            max_distance = 0
            selected_node = None

            for node_id in cluster_nodes:
                distance_sum = 0

                for other_node_id in cluster_nodes:
                    if node_id != other_node_id:
                        other_node = self.nodes[other_node_id]
                        dist_sqrd = (self.nodes[node_id].loc[0] - other_node.loc[0]) ** 2 + (
                                self.nodes[node_id].loc[1] - other_node.loc[1]) ** 2
                        distance_sum += dist_sqrd

                if distance_sum > max_distance:
                    max_distance = distance_sum
                    selected_node = node_id

            cluster_heads.append(selected_node)

        return cluster_list, cluster_heads
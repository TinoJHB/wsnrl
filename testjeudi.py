import copy
import random
import time
import numpy as np
import networkx as nx
from SACD import actor
from kmeansjeudi import ElbowMethodKMeans
from networktopologyjeudi import NetworkTopology
from EdgeServer import EdgeServer

G = nx.Graph()
gamma = 0.5

class MainEnvironment:
    def __init__(self, reward_function, state_dim, action_dim, num_agents, state_cluster):
        self.num_nodes = num_nodes  # Assign self.num_nodes as the length of the nodes_list
        self.num_clusters = num_clusters
        self.node_range = node_range
        self.m = m
        self.nodes_dict = {}
        self.current_step = 0
        self.max_steps = max_steps  # Define the maximum number of steps in an episode
        self.max_clusters = max_clusters
        self.cluster_heads = []
        self.prev_action = prev_action
        self.prev_state = prev_state
        self.prev_reward = prev_reward
        # energy levels, and RSSI values of the nodes
        self.E_levels = np.random.rand(num_nodes) * 100
        self.RSSIs = np.random.rand(num_nodes) * (-100)
        self.locs = np.array([random.uniform(0, 1), random.uniform(0, 1)])

        self.topology = NetworkTopology(E_level=self.E_levels, RSSI=self.RSSIs, comm_range=2.5, int_range=1.5,
                                        loc=self.locs, m=self.m, max_iterations=100, num_nodes=num_nodes)
        self.nodes = self.topology.generate_random_nodes(num_nodes)
        self.action = action
        self.reward = reward
        self.num_nodes_per_cluster = num_nodes_per_cluster
        self.clustered_dict = clustered_dict
        self.actor = actor
        self.reward = reward_function
        # Create the initial state of the environment
        self.state = []
        # Define the action and observation spaces
        self.action_space = self.create_action_space()
        self.observation_space = self.create_observation_space()
        # Use the form_clusters method of the NetworkTopology instance to generate the initial clusters
        self.clusters = self.topology.form_clusters(num_clusters)
        print(type(self.clusters))
        self.kmeans = ElbowMethodKMeans(max_clusters=10, max_iterations=100, final_model=None)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_cluster = state_cluster
        self.selected_node_indices = []
        # Create an EdgeServer instance
        self.edge_server_agent = EdgeServer(state_dim, action_dim, state_cluster)
        # Initialize the current state, action, and reward
        self.current_state = None
        self.current_action = None
        self.current_reward = None

###cette partie est a revoir
    def create_initial_state(self):
        state = []
        for i in range(self.num_nodes):
            node_state = []
            for j in range(self.num_clusters):
                if i in self.clusters[j]:
                    if j in self.clusters:
                        # Add the energy level and RSSI value of the node to the node state
                        node_state.append((self.E_levels[self.clusters[i][j]], self.RSSIs[self.clusters[i][j]]))
            state.append(node_state)
        return state

    def create_observation_space(self):
        # Define the observation space of the environment
        observation_space = [
            (self.num_clusters, self.num_nodes_per_cluster, 2)
            for _ in range(self.num_clusters * self.num_nodes_per_cluster)
        ]
        return observation_space

    def create_action_space(self):
        # Define the action space of the environment
        action_space = [(i, j, 2) for i in range(self.num_clusters)
                        for j in range(self.num_nodes_per_cluster)]
        return action_space
###cette partie est a revoir

    def step(self, selected_action):
        # Update the state of the environment based on the selected action
        self.state = self.create_initial_state()

        # Compute observation space state
        observation_space_state = self.state

        # Compute reward value
        reward_value = self.reward_function(self.state, selected_action)

        # Compute done flag
        done_flag = (self.current_step >= self.max_steps)

        # Update current step count
        self.current_step += 1

        # Return the new state, reward, and done flag
        return observation_space_state, reward_value, done_flag

    def reset(self):
        self.locs = np.array(self.topology.nodes)
        self.E_levels = np.random.rand(self.num_nodes) * 100
        self.RSSIs = np.random.rand(self.num_nodes) * (-100)
        self.clusters = [[] for _ in range(self.num_clusters)]
        self.state = []
        self.clustered_dict = {i: [] for i in range(self.num_clusters)}  # Initialize with empty lists

        # Populate the `clustered_dict` dictionary with the desired number of nodes per cluster
        node_indices = list(range(self.num_nodes))
        random.shuffle(node_indices)
        for i in range(self.num_clusters):
            cluster_indices = node_indices[i * self.num_nodes_per_cluster:(i + 1) * self.num_nodes_per_cluster]
            self.clustered_dict[i] = cluster_indices

        # Assign values to `self.clusters` using the `clustered_dict` dictionary
        for i in range(self.num_clusters):
            cluster_indices = self.clustered_dict[i]
            selected_node_indices = random.sample(list(cluster_indices), self.num_nodes_per_cluster)
            self.clusters[i].extend(selected_node_indices)
            state_cluster = []
            for j in selected_node_indices:
                node = {'Loc': self.nodes[j], 'E': self.E_levels[j], 'RSSI': self.RSSIs[j]}
                state_cluster.append(node)
            self.state.append(state_cluster)

    def reward_function(self, state, selected_action):
        # Calculate remaining energy of selected cluster head
        E_a = state[selected_action]['E']

        # Calculate average distance between selected cluster head and other nodes
        distances = []
        for i in range(len(state)):
            if i != selected_action:
                x1, y1 = state[selected_action]['Loc']
                x2, y2 = state[i]['Loc']
                d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distances.append(d)

        D_a = sum(distances) / len(distances)

        # Calculate RSSI of selected cluster head
        R_a = state[selected_action]['RSSI']

        # Assign weights to balance importance of energy, distance and RSSI metrics
        w1 = 0.5  # can be changed
        w2 = 0.3  # can be changed
        w3 = 0.2  # can be changed

        # Calculate overall reward based on weighted sum of metrics
        reward_value = w1 * E_a + w2 * D_a + w3 * R_a

        return reward_value

    def select_action(self, state_cluster):
        """Selects the best cluster head using the SACD algorithm with an edge server as the agent."""
        # Initialize variables for the best RSSI value and residual energy
        best_rssi = -float('inf')
        best_energy = -float('inf')
        # Initialize variable for the index of the selected cluster head
        selected_cluster_head = -1

        # Loop through each cluster
        for cluster_nodes in clusters:
            # Loop through each node in the cluster
            for node_id in cluster_nodes:
                # Check if the node ID is already in the `selected_node_indices` list
                if node_id not in self.selected_node_indices:
                    continue

                # Get the index of the node in the `selected_node_indices` list
                node_index = self.selected_node_indices.index(node_id)

                # Get the RSSI value and residual energy for the node
                rssi = state_cluster[node_index]['RSSI']
                energy = state_cluster[node_index]['E']

                # Use the edge server agent to select an action (cluster head) based on the state
                action = EdgeServer.select_action(state_cluster[node_index], 4, deterministic=True) # 4 refers to the state dim

                # Update the best RSSI value, residual energy, and selected cluster head
                if rssi > best_rssi or (rssi == best_rssi and energy > best_energy):
                    best_rssi = rssi
                    best_energy = energy
                    selected_cluster_head = node_id

        # Return the index of the selected cluster head
        return selected_cluster_head

    def update_nodes_info(self, action, next_state):
        # Get the node ID for the current node
        node_id = self.current_state['id']

        # Check if the node ID is already in the `selected_node_indices` list
        if node_id not in self.selected_node_indices:
            self.selected_node_indices.append(node_id)

        # Check if the `state_cluster` list has an element for the current node
        if len(self.state_cluster) < len(self.selected_node_indices):
            # If the `state_cluster` list is too short, append empty dictionaries to it until it has an element for each node
            num_new_nodes = len(self.selected_node_indices) - len(self.state_cluster)
            for i in range(num_new_nodes):
                self.state_cluster.append({})

        # Update the state information for the current node in the `state_cluster` list
        node_index = self.selected_node_indices.index(node_id)
        self.state_cluster[node_index]['id'] = node_id
        self.state_cluster[node_index]['RSSI'] = self.current_state['RSSI']
        self.state_cluster[node_index]['E'] = self.current_state['E']
        self.state_cluster[node_index]['Loc'] = self.current_state['Loc']

        # Call the `update()` method of the edge server agent
        done = False # assuming the episode is not done
        self.edge_server_agent.agent.update(self.current_state, action, self.current_reward, next_state, done)

        # Update the current state, action, and reward
        self.current_state = next_state
        self.current_action = self.edge_server_agent.select_action(self.current_state, deterministic=True)
        self.current_reward = None # set the reward to None since it is not used in the update step

        # Update the `state_cluster` list for the edge server agent
        self.edge_server_agent.state_cluster = self.state_cluster

    def collect_experiences_from_all_nodes(self):
        all_experiences = []

        # Create a list of random nodes using the NetworkTopology class
        nodes = self.topology.generate_random_nodes(self.num_nodes)

        # Collect experiences from each cluster in the environment
        for cluster_id in range(self.num_clusters):
            # Initialize the experiences list for the current cluster
            experiences = []

            # Collect experiences from each node in the current cluster
            for node_id, node in enumerate(
                    nodes[cluster_id * self.num_nodes_per_cluster: (cluster_id + 1) * self.num_nodes_per_cluster]):
                # Print the node type and value for debugging
                print("Node type:", type(node))
                print("Node value:", node)

                # Collect experiences for the current node
                state = node.get_state()
                done = False
                while not done:
                    # Select an action for the current state using the node agent
                    action = edge_server_agent.select_action(state, state_cluster)

                    # Take the action and observe the next state and reward using the NetworkTopology class
                    next_state, reward, done = self.topology.update_state(action,
                                                                          node_id + cluster_id * self.num_nodes_per_cluster)

                    # Append the experience to the experiences list
                    experience = {"state": state, "action": action, "reward": reward, "next_state": next_state,
                                  "done": done}
                    experiences.append(experience)

                    # Update the current state for the next iteration
                    state = next_state

                # Update the state cluster tensor with the final state of the current node
                self.state_cluster[cluster_id][node_id] = state

            # Use the edge server agent for the current cluster to select an action for the edge server in that cluster
            edge_server_state = self.state_cluster[cluster_id][0]  # use the state of the first node in the cluster
            edge_server_action = self.edge_server_agent[cluster_id].select_action(edge_server_state, experiences)

            # Take the edge server action for the edge server in the current cluster and observe the next state and reward
            edge_server_id = (cluster_id + 1) * self.num_nodes_per_cluster - 1
            next_state, reward, done = self.topology.update_state(edge_server_action, edge_server_id)

            # Append the experience for the edge server to the experiences list for that cluster
            edge_server_experience = {"state": edge_server_state, "action": edge_server_action, "reward": reward,
                                      "next_state": next_state, "done": done}
            experiences.append(edge_server_experience)

            # Update the state cluster tensor with the final state of the current edge server
            self.state_cluster[cluster_id][-1] = next_state

            # Append the experiences for the current cluster to the all_experiences list
            all_experiences.append(experiences)

        return all_experiences
    def update_central_DRL(self, trained_DRL_models):
        """
        This function updates the central DRL model using trained models from all edge servers by FedAvg algorithm.

        Input: List of trained DRL models from different Edge Servers

        Output: Updated/Ensemble version of Central DRL Model
        """
        # Check if there are any edge servers to update the central DRL model
        if trained_DRL_models is None:
            return None
        num_edge_servers = len(trained_DRL_models)
        new_model = copy.deepcopy(trained_DRL_models[0])

        for layer_idx in range(len(new_model.layers)):
            new_weights = trained_DRL_models[0].layers[layer_idx].get_weights()

            for edge_server_idx in range(1, num_edge_servers):
                edge_server_weights = trained_DRL_models[edge_server_idx].layers[layer_idx].get_weights()

                new_weights = [(new_weights[i] + edge_server_weights[i]) / num_edge_servers for i in
                               range(len(new_weights))]

            new_model.layers[layer_idx].set_weights(new_weights)

        return new_model

    def update_edge_server_model(self, edge_server_models, central_DRL_model):
        """
        Updates the edge server models using experiences collected by the nodes in the cluster.
        Input:
        - edge_server_models: List of edge server DRL models
        - central_DRL_model: Updated central DRL model
        Output: None
        """
        if edge_server_models is not None:
            for edge_server_model in edge_server_models:
                # Collect experiences from all nodes in the cluster
                experiences = self.collect_experiences_from_all_nodes()

                # Extract the states, actions, rewards, next states, and dones from the experiences
                states = [experience[0] for experience in experiences]
                actions = [experience[1] for experience in experiences]
                rewards = [experience[2] for experience in experiences]
                next_states = [experience[3] for experience in experiences]
                dones = [experience[4] for experience in experiences]

                # Convert the states and next states to numpy arrays
                states = np.array(states)
                next_states = np.array(next_states)

                # Compute the target Q values using the central DRL model
                target_q_values = central_DRL_model.predict(next_states)
                target_q_values[dones] = 0
                target_q_values = rewards + (gamma * np.max(target_q_values, axis=-1))

                # Train the edge server model using the experiences and the target Q values
                edge_server_model.fit(states, actions, target_q_values)

        else:
            pass
    # handle the case where edge_server_models is None

    def data_transmission(self, nodes):
        num_nodes = []  # Get the number of nodes
        for node_id in range(len(num_nodes)):
            if nodes[node_id].has_data():
                neighbors = nodes[node_id].get_neighbors()
                for neighbor_id in neighbors:
                    if not nodes[neighbor_id].is_transmitting():
                        # Perform carrier sensing
                        if not nodes[neighbor_id].is_sensing_channel():
                            nodes[neighbor_id].sense_channel()

                        if nodes[neighbor_id].is_channel_idle():
                            # Wait for random backoff time
                            backoff_time = random.randint(0, nodes[neighbor_id].get_backoff_window())
                            time.sleep(backoff_time)

                            # Transmit data packet
                            if nodes[node_id].transmit_data():
                                nodes[neighbor_id].receive_data(nodes[node_id].transmit_data())

    def generate_state(self, node, cluster_head, node_range):
        """Generate the state for a node in the network."""
        state = []


        # Add the location of the node
        state.append(node['Loc'])

        # Add the RSSI values for each neighboring node
        for neighbor_id, neighbor in node['Neighbors'].items():
            if neighbor_id != cluster_head:
                distance_squared = (node['Loc'][0] - neighbor['Loc'][0]) ** 2 + (
                            node['Loc'][1] - neighbor['Loc'][1]) ** 2
                if distance_squared <= node_range ** 2:
                    state.append(neighbor['RSSI'])

        # Add the energy level of the node
        state.append(node['Energy'])

        return tuple(state)

    def run_simulation(self, total_superframe_periods, edge_server=None, trained_DRL_models=None):
        # Initialize the central DRL model
        central_DRL_model = None

        for superframe_period in range(total_superframe_periods):
            # Initialize the reward for each node to 0
            rewards = [0] * self.num_nodes

            # Control Period
            i = 0
            for state, cluster_head in zip(self.state, self.cluster_heads):
                prev_reward = self.reward_function(state, cluster_head)
                rewards[i] = prev_reward
                i += 1

                # Generate the state for the current period
                state = self.generate_state(self.nodes[i], self.cluster_heads[i], self.node_range)


                # Store the experience and share it with the edge server
                experience = (self.prev_state[i], self.prev_action[i], prev_reward, state)
                edge_server.add_experience(i, experience)

                # Select the action for the current period
                action = self.nodes[i].select_action(state, self.clusters[self.cluster_heads[i]], edge_server)

                # Map the action to a cluster head
                self.cluster_heads[i] = self.clusters[self.cluster_heads[i]][action]

                # Update the node information with the current experience
                self.update_nodes_info(i, state)

            # Collect experiences from all nodes and update edge server models
            self.collect_experiences_from_all_nodes()
            self.update_edge_server_model(edge_server, central_DRL_model)

            # Update the central DRL model based on the edge server models
            central_DRL_model = self.update_central_DRL(trained_DRL_models)

            # Local policy network updating
            for i in range(len(self.nodes)):
                self.nodes[i].update_local_model(self.index)

            # Data Transmission Period
            self.data_transmission(nodes)

            # Calculate performance metrics for the current period
            for i in range(self.num_nodes):
                curr_reward = self.reward_function(self.state[i], self.cluster_heads[i])
                rewards[i] += curr_reward


            # Print the average reward for the current superframe period
            avg_reward = np.mean(rewards)
            print("Average reward for superframe period", superframe_period, ":", avg_reward)


# Define the experimental setup
num_nodes = 100
num_clusters = 10
node_range = 50
max_steps = 100
m=10
max_clusters= 15
state_dim=4
action_dim=5
state_cluster=list(range(num_nodes))
edge_server_agent = EdgeServer(state_dim, action_dim, state_cluster)

cluster_heads=5
num_nodes_per_cluster=10
prev_action=None
prev_state=None
prev_reward=None
nodes = list(range(num_nodes))  # Convert num_nodes into a list
action=None
reward=None
clustered_dict={}
clusters={}

actor=actor

if __name__ == '__main__':

    env = MainEnvironment(reward_function=None, state_dim=4, action_dim=5, num_agents=None, state_cluster=state_cluster)

    env.reset()
    # Run the experiments using the run_simulation method
    env.run_simulation(100)
    env.create_initial_state()



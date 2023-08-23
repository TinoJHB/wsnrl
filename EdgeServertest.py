
from random import randint
import torch
from SACD import SACD_Agent
from utils import concatenate_experiences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiences:
    def __init__(self, state, action, reward, next_state, mask):
        self.state = state # convert the state tuple to a list
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.mask = mask


class EdgeServer(torch.nn.Module):
    def __init__(self, state_dim, action_dim, state_cluster):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_cluster = state_cluster

        self.SACD_Agent = SACD_Agent(state_dim, action_dim)

    def select_action(self, state, state_cluster, deterministic=True):
        # Use the state and self.state_cluster or state and state_cluster according to my case to update the SACD model
        self.SACD_Agent.update(state, state_cluster)

        # Select the best action (cluster head) based on the SACD model
        action = self.SACD_Agent.select_action(state, deterministic)

        # Return the selected action
        return action

    def train_actor(self, experiences):
        # Convert the experiences to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(device)
        actions = torch.LongTensor([exp.action for exp in experiences]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(device)
        masks = torch.FloatTensor([exp.mask for exp in experiences]).unsqueeze(1).to(device)

        # Concatenate the experiences into a single Experiences object
        experiences = Experiences(states, actions, rewards, next_states, masks)

        # Train the SACD model on the collected experiences and self.state_cluster
        self.SACD_Agent.train(experiences)

        # Train the SACD model again on the aggregated experiences from the FedAvg algorithm and self.state_cluster
        aggregated_experiences = self.aggregate_experiences()
        self.SACD_Agent.train(aggregated_experiences)  # je me dis que au lieu de SACD_Agent je dois mettre FedAvg

        # Return the trained model
        return self.SACD_Agent.actor

    def save(self, index, b_envname):
        torch.save(self.SACD_Agent.actor.state_dict(), "./model/sacd_actor_{}_{}.pth".format(index, b_envname))
        torch.save(self.SACD_Agent.q_critic.state_dict(), "./model/sacd_actor_{}_{}.pth".format(index, b_envname))

    def load(self, index, b_envname):
        self.SACD_Agent.actor.load_state_dict(torch.load("./model/sacd_actor_{}_{}.pth".format(index, b_envname)))
        self.SACD_Agent.q_critic.load_state_dict(torch.load("./model/sacd_critic_{}_{}.pth".format(index, b_envname)))

# Assuming you have already imported the required modules and defined the EdgeServer class

# Define the state_dim, action_dim, and state_cluster variables
state_dim = 4  # Replace this with the appropriate dimension of your state
action_dim = 5  # Replace this with the appropriate dimension of your action
num_clusters = 5
num_nodes = 100
state_cluster = [randint(0, num_clusters - 1) for _ in range(num_nodes)]


# Create an instance of the EdgeServer class
edge_server = EdgeServer(state_dim, action_dim, state_cluster)

# Optional: Train the EdgeServer using experiences (replace with your actual experiences)
experiences = [Experiences(state=[1.0, 1.0, 2.0, 3.0], action=0, reward=0.5, next_state=[2.0, 3.0, 4.0], mask=1.0)]  # Replace this with your list of experiences
edge_server.train_actor(experiences)

# Optional: Save the trained model (if needed)
edge_server.save(index=1, b_envname="my_environment")

# Optional: Load a previously trained model (if needed)
edge_server.load(index=1, b_envname="my_environment")

# Now, you can use the select_action method to get actions based on state and state_cluster
state = [0.75, 10.0, 20.0, -60]
selected_action = edge_server.select_action(state, state_cluster)

# Print the selected action
print("Selected action:", selected_action)

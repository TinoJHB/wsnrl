import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from buffer import replay_buffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=(3,3), stride=1, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64 + action_dim, 1)

    def forward(self, state, action):
        # Apply convolutional layers with ReLU activation and max pooling
        b, r, w = state.size()
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten output for input to fully connected layers
        x = x.view(b, -1)

        # Concatenate state and action and apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = torch.cat([x, action], dim=1)
        x = self.fc2(x)

        # Output a single Q-value for each action
        return x

class actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(actor, self).__init__()

		# Define convolutional layers
		self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 3, kernel_size=(3,3), stride=1, padding=1)

		# Define fully connected layers
		self.fc1 = nn.Linear(3 , 64)
		self.fc2 = nn.Linear(64, action_dim)

	def forward(self, x):
		# Apply convolutional layers with ReLU activation and max pooling
		b,c,r,w=x.size()
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, kernel_size=2, stride=2)
		# Flatten output for input to fully connected layers
		x = x.view(b, -1)
		print("x",x.size())

		# Apply fully connected layers with ReLU activation
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		# Apply Softmax function to produce probability distribution over actions
		probs = F.softmax(x, dim=1)
		print("prob",probs)
		return probs


class SACD_Agent(object):
	def __init__(self, state_dim, action_dim, alpha=0.0003, gamma=0.99, batch_size=64, adaptive_alpha=False):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.batch_size = batch_size
		self.ReplayBuffer=[]
		self.gamma = gamma
		self.tau = 1e-3 * alpha
		self.hidden_layers = nn.ModuleList()
		self.adaptive_alpha = adaptive_alpha


		self.actor = actor(state_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0003)

		self.q_critic = critic(state_dim, action_dim).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=0.0003)

		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False

		self.alpha = alpha
		self.adaptive_alpha = adaptive_alpha
		if adaptive_alpha:
			# We use 0.6 because the recommended 0.98 will cause alpha explosion.
			self.target_entropy = 0.6 * (-np.log(1 / action_dim))  # H(discrete)>0
			self.log_alpha = torch.tensor(np.log(alpha), dtype=float, requires_grad=True, device=device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=0.0003)

		self.H_mean = 0

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device) #from (s_dim,) to (1, s_dim)
			probs = self.actor(state)
			if deterministic:
				action = probs.argmax(-1).item()
			else:
				action = Categorical(probs).sample().item()
			return action

	def train(self, replay_buffer):
		s, a, r, s_next, dw = replay_buffer.sample(self.batch_size)

		# ------------------------------------------ Train Critic ----------------------------------------#
		'''Compute the target soft Q value'''
		with torch.no_grad():
			next_probs = self.actor(s_next)  # [b,a_dim]
			next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
			next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1,
							   keepdim=True)  # [b,1]
			target_Q = r + (1 - dw) * self.gamma * v_next

		'''Update soft Q net'''
		q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)  # [b,1]
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# ------------------------------------------ Train Actor ----------------------------------------#
		for params in self.q_critic.parameters():
			# Freeze Q net, so you don't waste time on computing its gradient while updating Actor.
			params.requires_grad = False

		probs = self.actor(s)  # [b,a_dim]
		log_probs = torch.log(probs + 1e-8)  # [b,a_dim]
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=True)  # [b,1]

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters():
			params.requires_grad = True

		# ------------------------------------------ Train Alpha ----------------------------------------#
		if self.adaptive_alpha:
			with torch.no_grad():
				self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
			alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp().item()

		# ------------------------------------------ Update Target Net ----------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, index, b_envname):
		torch.save(self.actor.state_dict(), "./model/sacd_actor_{}_{}.pth".format(index, b_envname))
		torch.save(self.q_critic.state_dict(), "./model/sacd_actor_{}_{}.pth".format(index, b_envname))

	def load(self, index, b_envname):
		self.actor.load_state_dict(torch.load("./model/sacd_actor_{}_{}.pth".format(index, b_envname)))
		self.q_critic.load_state_dict(torch.load("./model/sacd_critic_{}_{}.pth".format(index, b_envname)))

	def update(self, state, state_cluster):
		# Implement the update logic here
		# This method should update the SACD_Agent based on the provided state and state_cluster
		# You can access and update the necessary attributes of the SACD_Agent inside this method
		# For example:
		self.state = state
		self.state_cluster = state_cluster




import random
import numpy as np
import torch as t
if __name__ == "__main__":
	Agent = SACD_Agent(4, 5)
	buffer=replay_buffer(2000, [4,5], 5)

	x=t.randn(1,1,4,5)
	for i in range(50):
		state=t.randn(1,4,5)
		action=0
		reward=1
		next_state=t.randn(1,4,5)
		buffer.store_transition(state, action, reward, next_state, True)
	print("x",x.size())
	print("buffer",buffer.mem_cntr)

	states, actions, rewards, states_, _=buffer.sample(32)
	print("states, actions, rewards, states_, dones",states, actions, rewards, states_)
	action=Agent.select_action(x,True)
	print("action",action)
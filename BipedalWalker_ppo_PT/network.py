"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(NN, self).__init__()
		self.layer1 = nn.Linear(input_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, output_dim)

	def forward(self, n_observations):
		# Convert observation to tensor if it's a numpy array
		if isinstance(n_observations, np.ndarray):
			n_observations = torch.tensor(n_observations, dtype=torch.float)

		x = F.relu(self.layer1(n_observations))
		x = F.relu(self.layer2(x))
		return self.layer3(x)
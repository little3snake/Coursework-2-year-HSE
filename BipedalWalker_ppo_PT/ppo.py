import gymnasium as gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		self.env = env
		self.n_observations = env.observation_space.shape[0]
		self.n_actions = env.action_space.shape[0]
		self.actor = policy_class(self.n_observations, self.n_actions)                                                   # ALG STEP 1
		self.critic = policy_class(self.n_observations, 1)
		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

		# Initialize the covariance matrix for get_action(...), evaluate(...)
		self.cov_var = torch.full(size=(self.n_actions,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			'timesteps_current': 0,
			'iterations_current': 0,
			'batch_lens': [],
			'batch_rewards': [],    # episodic returns
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_episodes):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		timesteps_current = 0
		iterations_current = 0

		while iterations_current < total_episodes:                                                                       # ALG STEP 2
			# collecting batch data
			batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens = self.rollout()                     # ALG STEP 3
			# Calculate how many timesteps we collected this batch
			timesteps_current += np.sum(batch_lens)
			iterations_current += 1
			self.logger['timesteps_current'] = timesteps_current
			self.logger['iterations_current'] = iterations_current

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_observations, batch_actions)
			A_k = batch_discounted_rewards - V.detach()
			# Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of
			# our advantages and makes convergence much more stable and faster. One person added this because
			# solving was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# update network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, current_log_probs = self.evaluate(batch_observations, batch_actions)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# Subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				ratios = torch.exp(current_log_probs - batch_log_probs) # it's like pi(new)/pi(old), but faster
				# Calculate parts of clip loss
				clip_loss_1part = ratios * A_k
				clip_loss_2part = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k # ratios, min, max
				# Calculate actor and critic losses.
				# Negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing -f(x)  = maximising f(x)
				actor_loss = (-torch.min(clip_loss_1part, clip_loss_2part)).mean()
				critic_loss = nn.MSELoss()(V, batch_discounted_rewards)

				# Calculate gradients (weights) and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True) # for logger
				self.actor_optim.step()
				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()
				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training
			self._log_summary()

			# Save our model each save_freq iteration
			if iterations_current % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

		# make .gif after learning
		obs, _ = self.env.reset()
		frames = []
		total_reward = 0
		done = False
		while not done:
			frame = self.env.render()
			frames.append(frame)
			action, _ = self.get_action(obs)
			obs, reward, terminated, truncated, _ = self.env.step(action)
			total_reward += reward
			done = terminated or truncated
		return frames, total_reward

	def rollout(self):
		"""
			This is where we collect the fresh batch of data
			from simulation (on-policy algorythm).

			Parameters:
				None

			Return:
				batch_observations - Shape: (number of timesteps, dimension of observation)
				batch_actions - Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_discounted_rewards - the dicounted rewards of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		batch_observations = []
		batch_actions = []
		batch_log_probs = []
		batch_rewards = []
		batch_discounted_rewards = []
		batch_lens = []
		episode_rewards = []
		current_timestep = 0
		# many episodes --while < timesteps_per_batch
		while current_timestep < self.timesteps_per_batch:
			episode_rewards = [] # reset episode_rewards
			observation, _ = self.env.reset() # reset the env
			done = False
			# one episode
			for episode_timestep in range(self.max_timesteps_per_episode):
				current_timestep += 1
				batch_observations.append(observation)
				action, log_prob = self.get_action(observation)
				observation, reward, terminated, truncated, _ = self.env.step(action)
				done = terminated or truncated
				episode_rewards.append(reward)
				batch_actions.append(action)
				batch_log_probs.append(log_prob)
				if done:
					break

			batch_lens.append(episode_timestep + 1) # counting from 1 (not from 0)
			batch_rewards.append(episode_rewards)

		batch_observations = torch.tensor(batch_observations, dtype=torch.float)
		batch_actions = torch.tensor(batch_actions, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_discounted_rewards = self.compute_discounted_rewards(batch_rewards)                                                              # ALG STEP 4
		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rewards'] = batch_rewards
		self.logger['batch_lens'] = batch_lens

		return batch_observations, batch_actions, batch_log_probs, batch_discounted_rewards, batch_lens

	def compute_discounted_rewards(self, batch_rewards):
		"""
			Compute the discounted rewards of each timestep in a batch given the rewards.

			Parameters:
				batch_rewards - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_discounted_rewards - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# Shape: num timesteps per episode
		batch_discounted_rewards = []
		# Iterate through each episode
		for episode_rews in reversed(batch_rewards):
			discounted_reward = 0
			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return
			for rew in reversed(episode_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_discounted_rewards.insert(0, discounted_reward)

		batch_discounted_rewards = torch.tensor(batch_discounted_rewards, dtype=torch.float)
		return batch_discounted_rewards

	def get_action(self, observation):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				observation - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(observation)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM - video unavailable
		distribution = MultivariateNormal(mean, self.cov_mat)
		# Sample an action from the distribution
		action = distribution.sample()
		# Calculate the log probability for that action
		log_prob = distribution.log_prob(action)
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_observations, batch_actions):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_observations - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_actions - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape: same as batch_discounted_rewards
		V = self.critic(batch_observations).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# code like in get_action(...)
		mean = self.actor(batch_observations)
		distribution = MultivariateNormal(mean, self.cov_mat)
		log_probs = distribution.log_prob(batch_actions)
		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1200           # Max number of timesteps per episode
		self.n_updates_per_iteration = 15               # Number of times to update actor/critic per iteration
		self.learning_rate = 0.000429                   # Learning rate of actor optimizer
		self.gamma = 0.994                              # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.28                                # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		self.save_freq = 10                             # How often we save in number of iterations

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))


	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values
		# time
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		timesteps_current = self.logger['timesteps_current']
		iterations_current = self.logger['iterations_current']
		avg_episode_lens = np.mean(self.logger['batch_lens'])
		avg_episode_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in self.logger['batch_rewards']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
		# Round decimal places for more aesthetic logging messages
		avg_episode_lens = str(round(avg_episode_lens, 2))
		avg_episode_rewards = str(round(avg_episode_rewards, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))
		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{iterations_current} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_episode_lens}", flush=True)
		print(f"Average Episodic Return: {avg_episode_rewards}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps current: {timesteps_current}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rewards'] = []
		self.logger['actor_losses'] = []
import gymnasium as gym
import math
import matplotlib
import numpy as np
import random
from collections import deque,namedtuple
import matplotlib.pyplot as plt
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #for class DQN

import sys
import optuna
sys.modules["gym"] = gym
from gym import spaces
from gym.envs.box2d.lunar_lander import *

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def optimize_dqn(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    memory_size = trial.suggest_int("memory_size", 10000, 100000, step=10000)
    epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 20000)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 20)
    tau = trial.suggest_float("tau", 0.001, 0.1)

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
    memory = ReplayMemory(memory_size)

    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    num_episodes = 100
    total_reward_over_trials = 0

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(policy_net, target_net, memory, batch_size, gamma, optimizer)
            if t % target_update_interval == 0:
                soft_update(policy_net, target_net, tau)
            if done:
                break
        total_reward_over_trials += total_reward

    avg_reward = total_reward_over_trials / num_episodes
    return avg_reward

# implements memory for storing agent experiences (sequences of transitions between states) for DQN
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) # selects random items from memory with a size "batch_size"

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__() # initializes the parent class nn.Module
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x): # determines how data flows through the neural network.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


episode_durations = []
loss_counter = 0

# change weights
def optimize_model(policy_net, target_net, memory, batch_size, gamma, optimizer):
    # if there aren't enough memory
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # from [[state1, action1, next_state1, reward1]
    #  [state2, action2, next_state2, reward2] ...]
    # to [[state1,state2 ...], [action1, action2 ...], ...]
    batch = Transition(*zip(*transitions))
    # eg tensor([True, False, True, True, False, ...], dtype=torch.bool)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # tensor([[state1],
    #         [state2],
    #         [state3]])
    non_final_next_states = torch.cat([s.float() for s in batch.next_state
                                                if s is not None], dim=0)
    state_batch = torch.cat(batch.state).float()
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).float()
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute MSE loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping (restricting grads values (<=100))
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 70)
    optimizer.step()

def soft_update(policy_net, target_net, tau):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1.0 - tau)
    target_net.load_state_dict(target_net_state_dict)


study = optuna.create_study(direction="maximize")
study.optimize(optimize_dqn,n_trials=100)

print("Best hyperparameters found by Optuna:")
print(study.best_params)


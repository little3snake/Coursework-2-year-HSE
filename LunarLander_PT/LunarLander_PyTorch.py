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
#import optuna
sys.modules["gym"] = gym
from gym import spaces
from gym.envs.box2d.lunar_lander import *

env = gym.make('LunarLander-v3', render_mode = 'rgb_array')

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
writer = SummaryWriter('runs/LunarLander-v3')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

learning_rate = 0.00143
gamma = 0.9907
memory_size = 90000
batch_size =256
epsilon_start = 0.8306
epsilon_end = 0.0909
epsilon_decay = 4074
target_update_interval = 19
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 200
else:
    num_episodes = 60
tau = 0.06405

n_actions = env.action_space.n # for output layer
state, info = env.reset()
n_observations = len(state) # for input layer

policy_net = DQN(n_observations, n_actions).to(device) # 1st network - DQN
target_net = DQN(n_observations, n_actions).to(device) # 2nd network - TargetNetwork
target_net.load_state_dict(policy_net.state_dict()) # copy the Q form DQN to TN

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
memory = ReplayMemory(memory_size)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random() # generate random number from 0 to 1
    eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1. * steps_done / epsilon_decay) #formula for decresing epsilon
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row (indices - index of max elem)
            # found, so we pick action with the larger expected reward.
            # indices.view(1,1) change the tensor's size to 1*1
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # return random action for training
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
loss_counter = 0

# change weights
def optimize_model():
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

def soft_update(policy_net, target_net):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1.0 - tau)
    target_net.load_state_dict(target_net_state_dict)

def save_frames_as_gif(frames, path='./', filename='cart_pole.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

frames = [] # for .gif

#training
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    for t in count():# episode duration
        if (i_episode == num_episodes - 1):
            frames.append(env.render())
        action = select_action(state)
        # .item() - translate form tensor to number
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        # from number to tensor
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        if (t % target_update_interval == 0):
            soft_update(policy_net, target_net)
        if done:
            episode_durations.append(t + 1)
            print(f"Episode {i_episode}: Total Reward: {total_reward}")
            writer.add_scalar('Episode reward', total_reward, i_episode)
            writer.add_scalar('Episode duration', t+1, i_episode)
            writer.add_scalar('Epsilon', epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay), i_episode)
            break



writer.close()
env.close()
save_frames_as_gif(frames)
print('Training complete')

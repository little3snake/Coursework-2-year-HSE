import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from torch.utils.tensorboard import SummaryWriter

import torch # also for automatic differentiation (torch.autograd)
import torch.nn as nn #neural networks
import torch.optim as optim #optimization
import torch.nn.functional as F
# create environment
env = gym.make("CartPole-v1")

# if GPU is to be used
# mps is Multy-Process Service (NVIDIA)
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

writer = SummaryWriter('runs/CartPole-v1')

# create simplified immutable objects (similar to structures) with named fields
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# implements memory for storing agent experiences (sequences of transitions between states) for DQN
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # two-way queue with maximum length "capacity"

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) # selects random items from memory with a size "batch_size"
        # for learning

    def __len__(self):
        return len(self.memory) # returns the length of the memory list

# In effect, the network is trying to predict the expected return of taking each action given the current input.
class DQN(nn.Module):
    # n_observation is the size of the input vector (the dimension of the state of the environment)
    # n_actions is the size of the output vector (number of actions)
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__() # initializes the parent class nn.Module
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x): # determines how data flows through the neural network.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # a vector of size n_actions, where each value is the Q—value for the corresponding action.

# Hyperparameters
BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.05 # EPS_END is the final value of epsilon
EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network (not DQN)
LR = 1e-4 # LR is the learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space
n_actions = env.action_space.n # for output layer
# Get the number of state observations
state, info = env.reset()
n_observations = len(state) # count the number of signs of state

policy_net = DQN(n_observations, n_actions).to(device) # 1st network - DQN
target_net = DQN(n_observations, n_actions).to(device) # 2nd network - TargetNetwork
target_net.load_state_dict(policy_net.state_dict()) # copy the Q form DQN to TN

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random() # genetare random number from 0 to 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY) #formula for decresing eps
    steps_done += 1
    if sample > eps_threshold:
        # Disables automatic gradient tracking in PyTorch
        # so that you can simply use the network to predict an action rather than train it.
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # indices.view(1,1) change the tensor's size to 1*1
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # return random action for training
        # torch.long trancate sample data to int
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

loss_counter = 0
# training
def optimize_model():
    # if there aren't enough memory
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # transition to a tuple of arrays,
    # where each array contains elements of the same type (for example, states, actions, rewards, etc.).
    # zip transpose list of transitions to suitable format for Batch
    # [[state1, action1, next_state1, reward1]
    #  [state2, action2, next_state2, reward2] ...]
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # eg tensor([True, False, True, True, False, ...], dtype=torch.bool)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # eg batch.next_state = [state1, None, state2, state3, None] -> [state1, state2, state3]
    # if tensor batch was (4,) after cat -> tensor (3,4)
    # tensor([[state1],
    #         [state2],
    #         [state3]])
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # for decresing storage consuming because we haven't to calculate gradients.
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss. Like a parabola
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping (restricting grads values (<=100))
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # to Tensorboard
    writer.add_scalar('loss', loss.item(), steps_done)


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # [...] -row
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

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            writer.add_scalar('Episode reward', total_reward, i_episode)
            writer.add_scalar('Episode duration', t+1, i_episode)
            writer.add_scalar('Epsilon', EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY), i_episode)
            break

writer.close()
print('Training complete')


import gymnasium as gym
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

from ppo import PPO
from network import NN

# we don't use it
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def save_frames_as_gif(frames, path='./', filename='bipedalWalker_ppo_PT_post_training.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

frames = [] # for .gif

def train(env, hyperparameters):
  """
    Trains the model.

    Parameters:
      env - the environment to train on
      hyperparameters - a dict of hyperparameters to use, defined in main

    Return:
      None
  """
  print(f"Training", flush=True) # immediate printing
  # Create a model for PPO.
  model = PPO(policy_class=NN, env=env, **hyperparameters)
  # Train the PPO model with a specified total episodes
  frames, total_reward = model.learn(total_episodes=400)
  save_frames_as_gif(frames)
  print ('total reward trained model= ', total_reward)

hyperparameters = {
      'timesteps_per_batch': 4096,
      'max_timesteps_per_episode': 1200,
      'gamma': 0.99421,
      'n_updates_per_iteration': 18,
      'learning_rate': 0.0004755,
      'clip': 0.286,
      }

# if you want to change env it must inherit Gym and have both continuous
# observation and action spaces.
env = gym.make("BipedalWalker-v3", render_mode='rgb_array')
train(env=env, hyperparameters=hyperparameters)


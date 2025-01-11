import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.results_plotter import ts2xy, load_results

from TimingCallback import TimingCallback


def save_frames_as_gif(frames, path='./', filename='bipedal_walker.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=60)


def record_episode_gif(env, model, filename):
    obs, _ = env.reset()
    frames = []
    total_reward = 0
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        done = done or truncated

    env.close()
    save_frames_as_gif(frames, path=video_dir, filename=filename)
    print(f"Total reward for {filename}: {total_reward}")

if __name__ == "__main__":
    log_dir = r"C:\Users\user\Python ML\PyTorch\BipedalWalker_ppo_PTorch\logs_ppo\ppo_sb_bipedal_walker_logs"
    os.makedirs(log_dir, exist_ok=True)
    video_dir = "./gif"
    os.makedirs(video_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    nn_layers = [128, 128]
    env_name = "BipedalWalker-v3"
  
    #env = gym.make(env_name)
    env = make_vec_env(env_name, n_envs=8, vec_env_cls=SubprocVecEnv)
    #env = Monitor(env, log_dir)
    #env = DummyVecEnv([lambda: env])
  
    timing_callback = TimingCallback(log_interval=10000, verbose=1) #  -- timing callback
    callback = EvalCallback(env, log_path=log_dir, deterministic=True)

    policy_kwargs = dict( ortho_init=False, activation_fn=torch.nn.ReLU,
                  net_arch=nn_layers )

    model = PPO("MlpPolicy", env,
                device="cpu", # for vectors
                learning_rate = 0.000329,
                n_steps = 2048,
                batch_size = 64,
                n_epochs = 10,
                gamma = 0.995,
                gae_lambda = 0.8813,
                clip_range = 0.15783,
                ent_coef = 0.0142,
                policy_kwargs = policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir)
    # via optuna:
    #Best hyperparameters:  {'learning_rate': 0.00032980794659323646, 'n_steps': 2048, 'batch_size': 64,
    # 'n_epochs': 10, 'gamma': 0.9950056046983321, 'gae_lambda': 0.8813326899122579,
    # 'clip_range': 0.1578379894239113, 'ent_coef': 0.014207080406496674}

    #pretraining
    #record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_pretraining.gif")

    total_timesteps = 500_000 # 500_000
    model.learn(total_timesteps=total_timesteps,
                log_interval=10,
                callback=[callback, timing_callback])

    # logs to tensorboard
    from stable_baselines3.common.results_plotter import ts2xy, load_results
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    for i in range(len(x)):
        writer.add_scalar("Episode Reward", y[i], x[i])
    writer.flush()

    record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_post_training.gif")

    writer.close()

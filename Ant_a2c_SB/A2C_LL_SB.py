import os
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.results_plotter import ts2xy, load_results

# Создание директорий для логов и видео
log_dir = "../Ant_a2c_PTorch/logs_a2c/a2c_sb_ant_logs"
os.makedirs(log_dir, exist_ok=True)
video_dir = "./gif"
os.makedirs(video_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir)

# Настройки сети и окружения
nn_layers = [64, 64]
env_name = "Ant-v5"
env = gym.make(env_name)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
callback = EvalCallback(env, log_path=log_dir, deterministic=True)

# Гиперпараметры A2C
policy_kwargs = dict(
    ortho_init=False,
    activation_fn=torch.nn.ReLU,
    net_arch=nn_layers
)

model = A2C(
    "MlpPolicy", env,
    learning_rate=0.003619,
    n_steps=256,
    gamma=0.909,
    gae_lambda=0.84,
    ent_coef=0.000716,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda"
)

# Функции для записи эпизодов в GIF
def save_frames_as_gif(frames, path='./', filename='ant.gif'):
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

# Дообучение (pretraining)
record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_pretraining.gif")

# Обучение модели
total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)

# Логирование результатов в TensorBoard
x, y = ts2xy(load_results(log_dir), 'timesteps')
for i in range(len(x)):
    writer.add_scalar("Episode Reward", y[i], x[i])
writer.flush()

# После обучения (post-training)
record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_post_training.gif")

writer.close()

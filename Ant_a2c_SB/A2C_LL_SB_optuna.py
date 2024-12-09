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
import optuna

# Создание директорий для логов и видео
log_dir = r"C:\Users\user\Python ML\PyTorch\LunarLander_a2c_PTorch\logs_a2c\a2c_sb_ant_logs"
os.makedirs(log_dir, exist_ok=True)
video_dir = "./gif"
os.makedirs(video_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir)

# Настройки окружения
env_name = "Ant-v5"
env = gym.make(env_name)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])


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
    reset_output = eval_env.reset()
    if isinstance(reset_output, tuple):
        obs, _ = reset_output  # Для новых версий gymnasium
    else:
        obs = reset_output  # Для старых версий gym
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


# Функция для оптимизации
def optimize_a2c(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 1.0)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-4, 1e-2)
    net_arch = trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]])

    # Конфигурация политики
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=net_arch,
        ortho_init=False,
    )

    # Создание модели
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_dir,
        device="cuda",
    )

    # Обучение модели
    model.learn(total_timesteps=100_000, log_interval=10)

    # Оценка производительности
    eval_env = DummyVecEnv([lambda: gym.make(env_name)])
    total_reward = 0
    episodes = 5
    for _ in range(episodes):
        reset_output = eval_env.reset()
        if isinstance(reset_output, tuple):
            obs, _ = reset_output  # Для новых версий gymnasium
        else:
            obs = reset_output  # Для старых версий gym
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward
            done = done

    avg_reward = total_reward / episodes
    return avg_reward


# Запуск Optuna
study = optuna.create_study(direction="maximize")
study.optimize(optimize_a2c, n_trials=40)

# Лучшие гиперпараметры
print("Best hyperparameters:", study.best_params)

# Использование лучших гиперпараметров для обучения модели
best_params = study.best_params
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=best_params["net_arch"],
    ortho_init=False,
)

model = A2C(
    "MlpPolicy",
    env,
    learning_rate=best_params["learning_rate"],
    n_steps=best_params["n_steps"],
    gamma=best_params["gamma"],
    gae_lambda=best_params["gae_lambda"],
    ent_coef=best_params["ent_coef"],
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir,
    device="cuda",
)

# Дообучение (pretraining)
record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_pretraining.gif")

# Полное обучение модели
total_timesteps = 500_000
model.learn(total_timesteps=total_timesteps, log_interval=10)

# Логирование результатов в TensorBoard
x, y = ts2xy(load_results(log_dir), "timesteps")
for i in range(len(x)):
    writer.add_scalar("Episode Reward", y[i], x[i])
writer.flush()

# После обучения (post-training)
record_episode_gif(gym.make(env_name, render_mode="rgb_array"), model, f"{env_name}_post_training.gif")

writer.close()

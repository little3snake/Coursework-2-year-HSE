import os
import torch
import gymnasium as gym
import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import ts2xy, load_results

log_dir = "./ppo_bipedal_walker_logs"
os.makedirs(log_dir, exist_ok=True)

env_name = "BipedalWalker-v3"
env = gym.make(env_name)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])

def objective(trial):
    # params for optimization
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    n_epochs = trial.suggest_int('n_epochs', 3, 10)
    gamma = trial.suggest_uniform('gamma', 0.98, 0.999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.95)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.02)

    policy_kwargs = dict(
        ortho_init=False,
        activation_fn=torch.nn.ReLU,
        net_arch=[128, 128]
    )

    model = PPO(
        "MlpPolicy", env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_dir
    )

    # Training model
    model.learn(total_timesteps=170000)

    # Evaluation
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(y) == 0:
        return -np.inf  # If can't find the data, return the lowest value
    avg_reward = np.mean(y[-10:])  # Mean reward for last 10 episodes
    return avg_reward

# Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # 30 iterations

print("Best hyperparameters: ", study.best_params)

# final training after finding optimal hyperparameters
best_params = study.best_params
model = PPO(
    "MlpPolicy", env,
    learning_rate=best_params['learning_rate'],
    n_steps=best_params['n_steps'],
    batch_size=best_params['batch_size'],
    n_epochs=best_params['n_epochs'],
    gamma=best_params['gamma'],
    gae_lambda=best_params['gae_lambda'],
    clip_range=best_params['clip_range'],
    ent_coef=best_params['ent_coef'],
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir
)

model.learn(total_timesteps=300000, log_interval=10, callback=EvalCallback(env, log_path=log_dir, deterministic=True))


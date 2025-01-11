from stable_baselines3.common.callbacks import BaseCallback
import time

class TimingCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super(TimingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.episode_start_time = None
        self.last_logged_timestep = 0

    def _on_step(self) -> bool:
        # Check log_interval steps
        current_timestep = self.num_timesteps
        if current_timestep - self.last_logged_timestep >= self.log_interval:
            elapsed_time = time.time() - self.start_time
            if self.verbose > 0:
                print(f"Time for {self.log_interval} steps: {elapsed_time:.2f} seconds")
            self.last_logged_timestep = current_timestep
            self.start_time = time.time()  # Restart timer
        return True

    def _on_training_start(self) -> None:
        # Save start time
        self.start_time = time.time()
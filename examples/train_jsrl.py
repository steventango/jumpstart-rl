import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from jsrl import get_jsrl_algorithm


def main():
    max_horizon = 150
    env = gym.make("PointMaze_UMaze-v3", continuing_task=False, max_episode_steps=max_horizon)
    guide_policy = SAC.load("models/sac_pointmaze_guide/best_model").policy
    n = 10
    model = get_jsrl_algorithm(SAC)(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            horizons=np.arange(max_horizon, -1, -max_horizon // n,),
            tolerance=0.0,
        ),
        verbose=1,
        tensorboard_log="logs"
    )
    model.learn(
        total_timesteps=1e6,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            best_model_save_path="models/sac_pointmaze_jsrl"
        ),
    )


if __name__ == "__main__":
    main()

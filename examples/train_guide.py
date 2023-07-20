import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


def main():
    env = gym.make("PointMaze_UMaze-v3", continuing_task=False, max_episode_steps=150)
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs"
    )
    model.learn(
        total_timesteps=1e5,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            best_model_save_path="examples/models/sac_pointmaze_guide"
        ),
    )


if __name__ == "__main__":
    main()

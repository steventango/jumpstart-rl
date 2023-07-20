from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("PointMaze_UMaze-v3")
model = SAC.load("sac_pointmaze_jsrl/best_model")
print(evaluate_policy(model, env, n_eval_episodes=10))

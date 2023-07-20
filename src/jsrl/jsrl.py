from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback


class JSRLAfterEvalCallback(BaseCallback):
    def __init__(self, policy, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.logger = logger
        self.best_moving_mean_reward = -np.inf
        self.mean_rewards = np.full(3, -np.inf, dtype=np.float32)

    def _on_step(self) -> bool:
        self.logger.record("eval/horizon", self.policy.horizons[self.policy.horizon_step])
        self.mean_rewards = np.roll(self.mean_rewards, 1)
        self.mean_rewards[0] = self.parent.last_mean_reward
        moving_mean_reward = np.mean(self.mean_rewards)
        self.logger.record("eval/moving_mean_reward", moving_mean_reward)
        if self.mean_rewards[-1] == -np.inf:
            return
        elif self.best_moving_mean_reward == -np.inf:
            self.best_moving_mean_reward = moving_mean_reward
        elif 1 - moving_mean_reward / self.best_moving_mean_reward <= self.policy.tolerance:
            horizon = self.policy.update_horizon()
            # self.mean_rewards = np.full(3, -np.inf, dtype=np.float32)
            if self.verbose:
                print(f"Updating horizon to {horizon}!")
        self.best_moving_mean_reward = max(self.best_moving_mean_reward, moving_mean_reward)


def get_jsrl_policy(ExplorationPolicy: BasePolicy):
    class JSRLPolicy(ExplorationPolicy):
        def __init__(
            self,
            *args,
            guide_policy: BasePolicy = None,
            max_horizon: int = 0,
            horizons: List[int] = [0],
            tolerance: float = None,
            strategy: str = "curriculum",
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.guide_policy = guide_policy
            self.tolerance = tolerance
            self.strategy = strategy
            self.horizon_step = 0
            self.max_horizon = max_horizon
            self.horizons = horizons

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            timesteps: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param timesteps: the number of timesteps since the beginning of the episode
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            timesteps_lte_horizon = timesteps <= self.horizons[self.horizon_step]
            timesteps_gt_horizon = timesteps > self.horizons[self.horizon_step]
            if isinstance(observation, dict):
                observation_lte_horizon = {k: v[timesteps_lte_horizon] for k, v in observation.items()}
                observation_gt_horizon = {k: v[timesteps_gt_horizon] for k, v in observation.items()}
            elif isinstance(observation, np.ndarray):
                observation_lte_horizon = observation[timesteps_lte_horizon]
                observation_gt_horizon = observation[timesteps_gt_horizon]
            if state is not None:
                state_lte_horizon = state[timesteps_lte_horizon]
                state_gt_horizon = state[timesteps_gt_horizon]
            else:
                state_lte_horizon = None
                state_gt_horizon = None
            if episode_start is not None:
                episode_start_lte_horizon = episode_start[timesteps_lte_horizon]
                episode_start_gt_horizon = episode_start[timesteps_gt_horizon]
            else:
                episode_start_lte_horizon = None
                episode_start_gt_horizon = None

            action = np.zeros((len(timesteps), *self.action_space.shape), dtype=self.action_space.dtype)
            if state is not None:
                state = np.zeros((len(timesteps), *state.shape[1:]), dtype=state_lte_horizon.dtype)

            if timesteps_lte_horizon.any():
                action_lte_horizon, state_lte_horizon = self.guide_policy.predict(
                    observation_lte_horizon, state_lte_horizon, episode_start_lte_horizon, deterministic
                )
                action[timesteps_lte_horizon] = action_lte_horizon
                if state is not None:
                    state[timesteps_lte_horizon] = state_lte_horizon

            if timesteps_gt_horizon.any():
                action_gt_horizon, state_gt_horizon = super().predict(
                    observation_gt_horizon, state_gt_horizon, episode_start_gt_horizon, deterministic
                )
                action[timesteps_gt_horizon] = action_gt_horizon
                if state is not None:
                    state[timesteps_gt_horizon] = state_gt_horizon

            return action, state

        def update_horizon(self) -> None:
            """
            Update the horizon based on the current strategy.
            """
            if self.strategy == "curriculum":
                self.horizon_step += 1
                self.horizon_step = min(self.horizon_step, len(self.horizons) - 1)
            elif self.strategy == "random":
                self.horizon_step = np.random.choice(self.max_horizon)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            return self.horizons[self.horizon_step]

    return JSRLPolicy


def get_jsrl_algorithm(Algorithm: BaseAlgorithm):
    class JSRLAlgorithm(Algorithm):
        def __init__(self, policy, *args, **kwargs):
            if isinstance(policy, str):
                policy = self._get_policy_from_name(policy)
            else:
                policy = policy
            policy = get_jsrl_policy(policy)
            super().__init__(policy, *args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)

        def _init_callback(
            self,
            callback: MaybeCallback,
            progress_bar: bool = False,
        ) -> BaseCallback:
            """
            :param callback: Callback(s) called at every step with state of the algorithm.
            :param progress_bar: Display a progress bar using tqdm and rich.
            :return: A hybrid callback calling `callback` and performing evaluation.
            """
            callback = super()._init_callback(callback, progress_bar)
            callback = CallbackList(
                [
                    callback,
                    EvalCallback(
                        self.env,
                        callback_after_eval=JSRLAfterEvalCallback(
                            self.policy,
                            self.logger,
                            verbose=self.verbose,
                        ),
                        eval_freq=1000,
                        n_eval_episodes=10,
                    ),
                ]
            )
            callback.init_callback(self)
            return callback

        def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            """
            Get the policy action from an observation (and optional hidden state).
            Includes sugar-coating to handle different observations (e.g. normalizing images).

            :param observation: the input observation
            :param state: The last hidden states (can be None, used in recurrent policies)
            :param episode_start: The last masks (can be None, used in recurrent policies)
                this correspond to beginning of episodes,
                where the hidden states of the RNN must be reset.
            :param deterministic: Whether or not to return deterministic actions.
            :return: the model's action and the next hidden state
                (used in recurrent policies)
            """
            action, state = self.policy.predict(observation, self._timesteps, state, episode_start, deterministic)

            self._timesteps += 1
            self._timesteps[self.env.buf_dones] = 0
            return action, state

    return JSRLAlgorithm

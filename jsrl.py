from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


def get_jsrl_policy(ExplorationPolicy: BasePolicy):
    class JSRLPolicy(ExplorationPolicy):
        def __init__(
            self,
            guide_policy: BasePolicy,
            *args,
            performance_threshold: float = 0.9,
            strategy: str = "curriculum",
            **kwargs
        ) -> None:
            super().__init__(*args, **kwargs)
            self.guide_policy = guide_policy
            self.performance_threshold = performance_threshold
            self.strategy = strategy

        # def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        #     """
        #     Forward pass in all the networks (actor and critic)

        #     :param obs: Observation
        #     :param deterministic: Whether to sample or use deterministic actions
        #     :return: action, value and log probability of the action
        #     """
        #     # Preprocess the observation if needed

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
            timesteps_lte_horizon = timesteps <= self.horizon
            timesteps_gt_horizon = timesteps > self.horizon
            if isinstance(observation, dict):
                observation_lte_horizon = {k: v[timesteps_lte_horizon] for k, v in observation.items()}
                observation_gt_horizon = {k: v[timesteps_gt_horizon] for k, v in observation.items()}
            elif isinstance(observation, np.ndarray):
                observation_lte_horizon = observation[timesteps_lte_horizon]
                observation_gt_horizon = observation[timesteps_gt_horizon]
            if state is not None:
                state_lte_horizon = [s[timesteps_lte_horizon] for s in state]
                state_gt_horizon = [s[timesteps_gt_horizon] for s in state]
            else:
                state_lte_horizon = None
                state_gt_horizon = None
            if episode_start is not None:
                episode_start_lte_horizon = episode_start[timesteps_lte_horizon]
                episode_start_gt_horizon = episode_start[timesteps_gt_horizon]
            else:
                episode_start_lte_horizon = None
                episode_start_gt_horizon = None

            action_lte_horizon, state_lte_horizon = self.guide_policy.predict(
                observation_lte_horizon, state_lte_horizon, episode_start_lte_horizon, deterministic
            )
            action_gt_horizon, state_gt_horizon = super().predict(
                observation_gt_horizon, state_gt_horizon, episode_start_gt_horizon, deterministic
            )
            action = np.concatenate((action_lte_horizon, action_gt_horizon))
            state = np.concatenate((state_lte_horizon, state_gt_horizon))
            return action, state

    return JSRLPolicy


def get_jsrl_algorithm(Algorithm: BaseAlgorithm):
    class JSRLAlgorithm(Algorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._timesteps = np.zeros((self.env.num_envs), dtype=np.int32)

        def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
            """
            Retrieve reward, episode length, episode success and update the buffer
            if using Monitor wrapper or a GoalEnv.

            :param infos: List of additional information about the transition.
            :param dones: Termination signals
            """
            super()._update_info_buffer(infos, dones)
            # Track episode timesteps
            if dones is not None:
                self._timesteps += 1
                self._timesteps[dones] = 0

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
            return self.policy.predict(observation, self._timesteps, state, episode_start, deterministic)

    return JSRLAlgorithm

from typing import Union, Optional, Callable
from pathlib import Path

from beartype import beartype
import numpy as np
import torch

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.clip_action import ClipAction
import envpool


BENCHMARKS = {
    "farama_mujoco": [
        f"{name}-v4" for name in
        [
            "Ant",
            "HalfCheetah",
            "Hopper",
            "HumanoidStandup",
            "Humanoid",
            "InvertedDoublePendulum",
            "InvertedPendulum",
            "Pusher",
            "Reacher",
            "Swimmer",
            "Walker2d",
        ]
    ],
    "deepmind_mujoco": [
        f"{name}-Feat-v0" for name in [
            "Hopper-Hop",
            "Cheetah-Run",
            "Walker-Walk",
            "Walker-Run",
            "Stacker-Stack_2",
            "Stacker-Stack_4",
            "Humanoid-Walk",
            "Humanoid-Run",
            "Humanoid-Run_Pure_State",
            "Humanoid_CMU-Stand",
            "Humanoid_CMU-Run",
            "Quadruped-Walk",
            "Quadruped-Run",
            "Quadruped-Escape",
            "Quadruped-Fetch",
            "Dog-Run",
            "Dog-Fetch",
        ]
    ],
}


@beartype
def get_benchmark(env_id: str) -> str:
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    return benchmark


@beartype
def make_env(env_id: str,
             seed: int,
             *,
             normalize_observations: bool,
             sync_vec_env: bool,
             num_envs: int,
             use_envpool: bool,
             video_path: Optional[Path] = None,
             horizon: Optional[int] = None,
             device: Optional[torch.device] = None,
    ) -> (tuple[Union[Env, SyncVectorEnv, AsyncVectorEnv],
          dict[str, tuple[int, ...]],
          np.ndarray,
          np.ndarray]):

    bench = get_benchmark(env_id)

    def make_env() -> Callable[[], Env]:
        def thunk() -> Env:
            if video_path is not None:
                assert sync_vec_env and (num_envs == 1)
                env = gym.make(env_id, render_mode="rgb_array")
                env = RecordVideo(env, str(video_path))
            elif use_envpool:
                env = envpool.make(
                    env_id,
                    env_type="gymnasium",
                    num_envs=num_envs,
                    seed=seed,
                )
                env.num_envs = num_envs
                env.single_action_space = env.action_space
                env.single_observation_space = env.observation_space
            else:
                env = gym.make(env_id)
                env = RecordEpisodeStatistics(env)
                env = ClipAction(env)
                if normalize_observations:
                    env = NormalizeObservation(env)
                    env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
                if horizon is not None:
                    env = TimeLimit(env, max_episode_steps=horizon)
            return env
        return thunk

    # create env
    if use_envpool:
        env = make_env()()
    else:
        env = (SyncVectorEnv if sync_vec_env else AsyncVectorEnv)(
            [
                make_env() for _ in range(num_envs)
            ],
        )
        env.action_space.seed(seed)  # to be fully reproducible

    # due diligence checks
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box)
    ac_space = env.action_space
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError("actions must be continuous")
    assert isinstance(ac_space, gym.spaces.Box)
    net_shapes = {"ob_shape": ob_space.shape, "ac_shape": ac_space.shape}

    # assert that all envs have the same action bounds
    assert np.all(ac_space.low == ac_space.low[0])
    assert np.all(ac_space.high == ac_space.high[0])

    return env, net_shapes, ac_space.low[0], ac_space.high[0]

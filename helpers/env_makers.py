from typing import Union, Optional, Callable
from pathlib import Path

from beartype import beartype
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.clip_action import ClipAction


# Farama Foundation Gymnasium MuJoCo
FARAMA_MUJOCO_STEM = [
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
FARAMA_MUJOCO = []
FARAMA_MUJOCO.extend([f"{name}-v4"
    for name in FARAMA_MUJOCO_STEM])

# DeepMind Control Suite (DMC) MuJoCo
DEEPMIND_MUJOCO_STEM = [
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
DEEPMIND_MUJOCO = []
DEEPMIND_MUJOCO.extend([f"{name}-Feat-v0"
    for name in DEEPMIND_MUJOCO_STEM])

# Flag benchmarks that are not available yet
BENCHMARKS = {"farama_mujoco": FARAMA_MUJOCO, "deepmind_mujoco": DEEPMIND_MUJOCO}
AVAILABLE_FLAGS = dict.fromkeys(BENCHMARKS, True)
AVAILABLE_FLAGS["deepmind_mujoco"] = False  # TODO(lionel): integrate with the DMC suite


@beartype
def get_benchmark(env_id: str):
    # verify that the specified env is amongst the admissible ones
    benchmark = None
    for k, v in BENCHMARKS.items():
        if env_id in v:
            benchmark = k
            continue
    assert benchmark is not None, "unsupported environment"
    assert AVAILABLE_FLAGS[benchmark], "unavailable benchmark"
    return benchmark


@beartype
def make_env(env_id: str,
             seed: int,
             *,
             normalize_observations: bool,
             sync_vec_env: bool,
             num_env: int,
             video_path: Optional[Path] = None,
             horizon: Optional[int] = None,
    ) -> (tuple[Union[SyncVectorEnv, AsyncVectorEnv],
          dict[str, tuple[int, ...]],
          np.ndarray,
          np.ndarray]):

    # to deal with dm_control's Dict observation space: env = gym.wrappers.FlattenObservation(env)

    bench = get_benchmark(env_id)

    if bench == "farama_mujoco":
        return make_farama_mujoco_env(env_id,
                                      seed,
                                      normalize_observations=normalize_observations,
                                      sync_vec_env=sync_vec_env,
                                      num_env=num_env,
                                      video_path=video_path,
                                      horizon=horizon)
    raise ValueError(f"invalid benchmark: {bench}")


@beartype
def make_farama_mujoco_env(env_id: str,
                           seed: int,
                           *,
                           normalize_observations: bool,
                           sync_vec_env: bool,
                           num_env: int,
                           video_path: Optional[Path] = None,
                           horizon: Optional[int] = None,
    ) -> (tuple[Union[SyncVectorEnv, AsyncVectorEnv],
          dict[str, tuple[int, ...]],
          np.ndarray,
          np.ndarray]):

    def make_env() -> Callable[[], Env]:
        def thunk() -> Env:
            if video_path is not None:
                assert sync_vec_env and (num_env == 1)
                env = gym.make(env_id, render_mode="rgb_array")
                env = RecordVideo(env, str(video_path))
            else:
                env = gym.make(env_id)
            env = FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = RecordEpisodeStatistics(env)
            env = ClipAction(env)
            if normalize_observations:
                env = NormalizeObservation(env)
                env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env.action_space.seed(seed)
            if horizon is not None:
                env = TimeLimit(env, max_episode_steps=horizon)
            return env
        return thunk

    # create env
    env = (SyncVectorEnv if sync_vec_env else AsyncVectorEnv)(
        [
            make_env() for _ in range(num_env)
        ],
    )

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

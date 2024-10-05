import os
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
from gymnasium.wrappers.clip_action import ClipAction

import envpool

from gymnasium.spaces import Box
from dm_control import suite
from dm_env import specs


BENCHMARKS = {
    "gym": [
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
    "dmcs": [
        f"{name}" for name in [
            "walker-walk",
            "humanoid_CMU-walk",
            # "Hopper-Hop",
            # "Cheetah-Run",
            # "Walker-Walk",
            # "Walker-Run",
            # "Stacker-Stack_2",
            # "Stacker-Stack_4",
            # "Humanoid-Walk",
            # "Humanoid-Run",
            # "Humanoid-Run_Pure_State",
            # "Humanoid_CMU-Stand",
            # "Humanoid_CMU-Run",
            # "Quadruped-Walk",
            # "Quadruped-Run",
            # "Quadruped-Escape",
            # "Quadruped-Fetch",
            # "Dog-Run",
            # "Dog-Fetch",
        ]
    ],
}


def _spec_to_box(spec):

    def extract_min_max(s):
        assert s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        if type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        raise TypeError("unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(np.float32)
    high = np.concatenate(maxs, axis=0).astype(np.float32)
    assert low.shape == high.shape
    return Box(low, high, dtype=np.float32)


@beartype
def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(np.float32)


class DeepMindControlSuite(Env):

    def __init__(self,
                 domain_name,
                 task_name,
                 /,
                 rendering="egl",
                 render_height=64,
                 render_width=64,
                 render_camera_id=0):

        # for details see https://github.com/deepmind/dm_control
        assert rendering in {"glfw", "egl", "osmesa"}
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(domain_name=domain_name, task_name=task_name)

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])

    def step(self, action):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = _flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed

        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)


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
                if bench == "dmcs":
                    domain, task = env_id.split("-")
                    env = DeepMindControlSuite(domain, task)
                    env.action_space = env._action_space
                    env.observation_space = env._observation_space
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

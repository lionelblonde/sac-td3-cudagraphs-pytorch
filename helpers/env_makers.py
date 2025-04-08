import os
from typing import Union, Optional, Callable, Iterable, Any
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
        f"{name}" for name in [  # leave like this
            "cartpole-swingup",  # this
            "hopper-hop",
            "walker-walk",  # this
            "walker-run",
            "cheetah-walk",
            "cheetah-run",  # this
            "humanoid-walk",  # this
            "humanoid-run",
            "humanoid_CMU-walk",
            "humanoid_CMU-run",
            "stacker-stack_2",
            "stacker-stack_4",
            "quadruped-walk",
            "quadruped-run",
            "quadruped-escape",
            "quadruped-fetch",
            "finger-spin",  # this
            "dog-run",
            "dog-fetch",
        ]
    ],
}


class DeepMindControlSuite(Env):
    """credit: https://github.com/imgeorgiev/dmc2gymnasium
    Not exact replica, but a subset of the above + type hints
    """

    @beartype
    def __init__(self,
                 domain_name: str,
                 task_name: str,
                 /,
                 rendering: str = "egl",
                 render_height: int = 64,
                 render_width: int = 64,
                 render_camera_id: int = 0):

        # for details see https://github.com/deepmind/dm_control
        assert rendering in {"glfw", "egl", "osmesa"}
        os.environ["MUJOCO_GL"] = rendering

        self._env = suite.load(domain_name=domain_name, task_name=task_name)

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id

        self._observation_space = self.spec_to_box(self._env.observation_spec().values())
        self._action_space = self.spec_to_box([self._env.action_spec()])

    @staticmethod
    @beartype
    def spec_to_box(spec: Iterable) -> Box:

        @beartype
        def extract_min_max(s: Union[specs.Array, specs.BoundedArray],
        ) -> tuple[np.ndarray, np.ndarray]:
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
        low = np.asarray(np.concatenate(mins, axis=0), dtype=np.float32)
        high = np.asarray(np.concatenate(maxs, axis=0), dtype=np.float32)
        assert low.shape == high.shape
        return Box(low, high, dtype=np.float32)

    @staticmethod
    @beartype
    def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v], copy=True) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0).astype(np.float32)

    @beartype
    def step(self,
             action: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        observation = self.flatten_obs(timestep.observation)
        reward = timestep.reward
        termination = False  # we never reach a goal
        truncation = timestep.last()
        info = {"discount": timestep.discount}
        return observation, reward, termination, truncation, info

    @beartype
    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict[str, Any]] = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
        assert options is None
        if seed is not None:
            seed_ = None
            if not isinstance(seed, np.random.RandomState):
                seed_ = np.random.RandomState(seed)
            self._env.task._random = seed_

        timestep = self._env.reset()
        observation = self.flatten_obs(timestep.observation)
        info = {}
        return observation, info

    @beartype
    def render(self,
               height: Optional[int] = None,
               width: Optional[int] = None,
               camera_id: Optional[int] = None) -> np.ndarray:
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
             video_path: Optional[Path] = None,
             horizon: Optional[int] = None,
    ) -> (tuple[Union[Env, SyncVectorEnv, AsyncVectorEnv],
          dict[str, tuple[int, ...]],
          np.ndarray,
          np.ndarray]):

    bench = get_benchmark(env_id)

    @beartype
    def make_env() -> Callable[[], Env]:
        @beartype
        def thunk() -> Env:
            if video_path is not None:
                assert sync_vec_env and (num_envs == 1)
                env = gym.make(env_id, render_mode="rgb_array")
                env = RecordVideo(env, str(video_path))
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

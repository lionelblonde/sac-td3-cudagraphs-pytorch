from typing import Union, Optional

from beartype import beartype
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from helpers import logger


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
def make_env(
    env_id: str,
    horizon: int,
    seed: int,
    *,
    vectorized: bool,
    multi_proc: bool,
    num_env: Optional[int] = None,
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], np.ndarray, np.ndarray]:

    # create an environment
    bench = get_benchmark(env_id)  # at this point benchmark is valid

    if bench == "farama_mujoco":
        return make_farama_mujoco_env(
            env_id,
            horizon,
            seed,
            vectorized=vectorized,
            multi_proc=multi_proc,
            num_env=num_env,
            record=record,
            render=render,
        )
    raise ValueError(f"invalid benchmark: {bench}")


@beartype
def make_farama_mujoco_env(
    env_id: str,
    horizon: int,
    seed: int,
    *,
    vectorized: bool,
    multi_proc: bool,
    num_env: Optional[int],
    record: bool,
    render: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], np.ndarray, np.ndarray]:

    # not ideal for code golf, but clearer for debug

    assert sum([record, vectorized]) <= 1, "not both same time"
    assert sum([render, vectorized]) <= 1, "not both same time"
    assert (not vectorized) or (num_env is not None), "must give num_envs when vectorized"

    def thunk(*, render_mode: Optional[str] = None):
        env = gym.make(env_id, render_mode=render_mode)
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    # create env
    # normally the windowed one is "human" .other option for later: "rgb_array", but prefer:
    # the following: `from gymnasium.wrappers.pixel_observation import PixelObservationWrapper`
    if record:  # overwrites render
        # assert horizon is not None
        # env = gym.make(env_id, render_mode="rgb_array")
        env = thunk(render_mode="rgb_array")
        # env = TimeLimit(gym.make(env_id, render_mode="rgb_array"), max_episode_steps=horizon)
    elif render:
        # assert horizon is not None
        # env = gym.make(env_id, render_mode="human")
        env = thunk(render_mode="human")
        # env = TimeLimit(gym.make(env_id, render_mode="human"), max_episode_steps=horizon)
    elif vectorized:
        assert num_env is not None
        # assert horizon is not None
        env = (AsyncVectorEnv if multi_proc else SyncVectorEnv)([
            lambda: thunk()
            # lambda: TimeLimit(gym.make(env_id), max_episode_steps=horizon)
            for _ in range(num_env)
        ])
        assert isinstance(env, (AsyncVectorEnv, SyncVectorEnv))
        logger.info("using vectorized envs")
    else:
        # assert horizon is not None
        # env = gym.make(env_id)
        env = thunk()
        # env = TimeLimit(gym.make(env_id), max_episode_steps=horizon)

    # build shapes for nets and replay buffer
    net_shapes = {}
    erb_shapes = {}

    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box)  # for due diligence
    ob_shape = ob_space.shape
    assert ob_shape is not None
    ac_space = env.action_space  # used now and later to get max action
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError(f"env ({env}) is discrete: out of scope here")
    assert isinstance(ac_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_shape = ac_space.shape
    assert ac_shape is not None
    net_shapes.update({"ob_shape": ob_shape, "ac_shape": ac_shape})

    erb_shapes.update({
        "obs0": (ob_shape[-1],),
        "acs0": (ac_shape[-1],),
        "obs1": (ob_shape[-1],),
        "erews1": (1,),
        "dones1": (1,),
    })
    min_ac, max_ac = ac_space.low, ac_space.high
    if vectorized:  # all envs have the same ac bounds
        min_ac, max_ac = ac_space.low[0], ac_space.high[0]
    return env, net_shapes, erb_shapes, min_ac, max_ac

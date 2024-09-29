from typing import Union, Optional, Callable

from beartype import beartype
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncVectorEnv


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
             sync_vec_env: bool,
             num_env: int,
             capture_video: bool,
    ) -> tuple[Union[Env, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], np.ndarray, np.ndarray]:

    # create an environment
    bench = get_benchmark(env_id)  # at this point benchmark is valid

    if bench == "farama_mujoco":
        return make_farama_mujoco_env(env_id,
                                      seed,
                                      sync_vec_env=sync_vec_env,
                                      num_env=num_env,
                                      capture_video=capture_video)
    raise ValueError(f"invalid benchmark: {bench}")


@beartype
def make_farama_mujoco_env(env_id: str,
                           seed: int,
                           *,
                           sync_vec_env: bool,
                           num_env: int,
                           capture_video: bool,
                           horizon: Optional[int] = None,
    ) -> tuple[Union[SyncVectorEnv, AsyncVectorEnv],
    dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], np.ndarray, np.ndarray]:

    def make_env(*, render_mode: Optional[str] = None) -> Callable[[], Env]:
        def thunk() -> Env:
            if render_mode is not None:
                env = gym.make(env_id, render_mode=render_mode)
            else:
                env = gym.make(env_id)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            if horizon is not None:
                env = TimeLimit(env, max_episode_steps=horizon)
            return env
        return thunk

    # create env
    if capture_video:
        env = make_env(render_mode="rgb_array")
        # env = RecordVideo(env, f"videos/{run_name}")
        env = SyncVectorEnv([env])
    else:
        env = (SyncVectorEnv if sync_vec_env else AsyncVectorEnv)(
            [
                make_env() for _ in range(num_env)
            ],
        )

    net_shapes = {}
    erb_shapes = {}

    # observations
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box)
    ob_shape = ob_space.shape
    assert ob_shape is not None

    # actions
    ac_space = env.action_space
    if isinstance(ac_space, gym.spaces.Discrete):
        raise TypeError(f"env ({env}) is discrete: out of scope here")
    assert isinstance(ac_space, gym.spaces.Box)

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
    # assert that all envs have the same action bounds
    assert np.all(min_ac == min_ac[0])
    assert np.all(max_ac == max_ac[0])
    # replace them with the bounds of the first env
    min_ac, max_ac = ac_space.low[0], ac_space.high[0]

    return env, net_shapes, erb_shapes, min_ac, max_ac

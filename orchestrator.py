import os
import time
import tqdm
from pathlib import Path
from typing import Union, Callable, Generator
from contextlib import contextmanager
from collections import deque

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import rearrange
from termcolor import colored
import wandb
from wandb.errors import CommError
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from gymnasium.core import Env
from gymnasium.vector.vector_env import VectorEnv

from helpers import logger
from agents.agent import Agent


@contextmanager
@beartype
def timed(op: str, timer: Callable[[], float]):
    logger.info(colored(
        f"starting timer | op: {op}",
        "magenta", attrs=["underline", "bold"]))
    tstart = timer()
    yield
    tot_time = timer() - tstart
    logger.info(colored(
        f"stopping timer | op took {tot_time}secs",
        "magenta"))


@beartype
def segment(env: Union[Env, VectorEnv],
            agent: Agent,
            seed: int,
            segment_len: int,
            learning_starts: int,
            action_repeat: int,
    ) -> Generator[None, None, None]:

    assert agent.rb is not None

    obs, _ = env.reset(seed=seed)  # for the very first reset, we give a seed (and never again)
    obs = torch.as_tensor(obs, device=agent.device, dtype=torch.float)
    actions = None  # as long as r is init at 0: ac will be written over

    t = 0
    r = 0  # action repeat reference

    while True:

        if r % action_repeat == 0:
            # predict action
            if agent.timesteps_so_far < learning_starts:
                actions = env.action_space.sample()
            else:
                actions = agent.predict(
                    TensorDict(
                        {
                            "observations": obs,
                        },
                        device=agent.device,
                    ),
                    explore=True,
                )

        if t > 0 and t % segment_len == 0:
            yield

        # interact with env
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        next_obs = torch.as_tensor(next_obs, device=agent.device, dtype=torch.float)
        real_next_obs = next_obs.clone()

        for idx, trunc in enumerate(np.array(truncations)):
            if trunc:
                real_next_obs[idx] = torch.as_tensor(
                    infos["final_observation"][idx], device=agent.device, dtype=torch.float)

        rewards = rearrange(
            torch.as_tensor(rewards, device=agent.device, dtype=torch.float),
            "b -> b 1",
        )
        terminations = rearrange(
            torch.as_tensor(terminations, device=agent.device, dtype=torch.bool),
            "b -> b 1",
        )

        agent.rb.extend(
            TensorDict(
                {
                    "observations": obs,
                    "next_observations": real_next_obs,
                    "actions": torch.as_tensor(actions, device=agent.device, dtype=torch.float),
                    "rewards": torch.as_tensor(rewards, device=agent.device, dtype=torch.float),
                    "terminations": terminations,
                    "dones": terminations,
                },
                batch_size=obs.shape[0],
                device=agent.device,
            ),
        )

        obs = next_obs

        t += 1
        r += 1


@beartype
def episode(env: Env,
            agent: Agent,
            seed: int,
            *,
            need_lists: bool = False,
    ) -> Generator[dict[str, np.ndarray], None, None]:
    # generator that spits out a trajectory collected during a single episode

    # `append` operation is significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible

    def randomize_seed() -> int:
        return seed + rng.integers(2**32 - 1, size=1).item()
        # seeded Generator: deterministic -> reproducible

    obs_list = []
    next_obs_list = []
    actions_list = []
    rewards_list = []
    terminations_list = []
    dones_list = []

    ob, _ = env.reset(seed=randomize_seed())
    if need_lists:
        obs_list.append(ob)
    ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)

    while True:

        # predict action
        action = agent.predict(
            TensorDict(
                {
                    "observations": ob,
                },
                device=agent.device,
            ),
            explore=False,
        )

        new_ob, reward, termination, truncation, infos = env.step(action)

        done = termination or truncation

        if need_lists:
            next_obs_list.append(new_ob)
            actions_list.append(action)
            rewards_list.append(reward)
            terminations_list.append(termination)
            dones_list.append(done)
            if not done:
                obs_list.append(new_ob)

        new_ob = torch.as_tensor(new_ob, device=agent.device, dtype=torch.float)
        ob = new_ob

        if "final_info" in infos:
            # we have len(infos["final_info"]) == 1
            for info in infos["final_info"]:
                ep_len = float(info["episode"]["l"].item())
                ep_ret = float(info["episode"]["r"].item())

            if need_lists:
                out = {
                    "observations": np.array(obs_list),
                    "actions": np.array(actions_list),
                    "next_observations": np.array(next_obs_list),
                    "rewards": np.array(rewards_list),
                    "terminations": np.array(terminations_list),
                    "dones": np.array(dones_list),
                    "length": np.array(ep_len),
                    "return": np.array(ep_ret),
                }
            else:
                out = {
                    "length": np.array(ep_len),
                    "return": np.array(ep_ret),
                }
            yield out

            if need_lists:
                obs_list = []
                next_obs_list = []
                actions_list = []
                rewards_list = []
                terminations_list = []
                dones_list = []

            ob, _ = env.reset(seed=randomize_seed())
            if need_lists:
                obs_list.append(ob)
            ob = torch.as_tensor(ob, device=agent.device, dtype=torch.float)


@beartype
def train(cfg: DictConfig,
          env: Union[Env, VectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], Agent],
          name: str):

    assert isinstance(cfg, DictConfig)

    agent = agent_wrapper()

    assert agent.rb is not None

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # save the model as a dry run, to avoid bad surprises at the end
    agent.save(ckpt_dir, sfx="dryrun")
    logger.info(f"dry run -- saved model @: {ckpt_dir}")

    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    group = ".".join(name.split(".")[:-1])  # everything in name except seed
    logger.warn(f"{name=}")
    logger.warn(f"{group=}")
    while True:
        try:
            config = OmegaConf.to_object(cfg)
            assert isinstance(config, dict)
            wandb.init(
                project=cfg.wandb_project,
                name=name,
                id=name,
                group=group,
                config=config,
                dir=cfg.root,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    # create segment generator for training
    seg_gen = segment(
        env, agent, cfg.seed, cfg.segment_len, cfg.learning_starts, cfg.action_repeat)
    # create episode generator for evaluating
    ep_gen = episode(eval_env, agent, cfg.seed)

    i = 0
    start_time = None
    measure_burnin = None
    pbar = tqdm.tqdm(range(cfg.num_timesteps))
    time_spent_eval = 0

    tlog = TensorDict({})
    maxlen = 20 * cfg.eval_steps
    len_buff = deque(maxlen=maxlen)
    ret_buff = deque(maxlen=maxlen)

    mode = None
    tc_update_actor = agent.update_actor
    tc_update_qnets = agent.update_qnets
    if cfg.compile:
        tc_update_actor = torch.compile(tc_update_actor, mode=mode)
        tc_update_qnets = torch.compile(tc_update_qnets, mode=mode)
    if cfg.cudagraphs:
        tc_update_actor = CudaGraphModule(tc_update_actor, in_keys=[], out_keys=[])
        tc_update_qnets = CudaGraphModule(tc_update_qnets, in_keys=[], out_keys=[])

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if ((agent.timesteps_so_far >= (cfg.measure_burnin + cfg.learning_starts) and
             start_time is None)):
            start_time = time.time()
            measure_burnin = agent.timesteps_so_far

        logger.info(("interact").upper())
        next(seg_gen)
        agent.timesteps_so_far += (increment := cfg.segment_len * cfg.num_envs)
        pbar.update(increment)

        if agent.timesteps_so_far <= cfg.learning_starts:
            # start training when enough data
            pbar.set_description("not learning yet")
            i += 1
            continue

        logger.info(("train").upper())

        # sample batch of transitions
        batch = agent.rb.sample(cfg.batch_size)

        # update qnets
        tlog.update(tc_update_qnets(batch))
        agent.qnet_updates_so_far += 1

        # update actor (and alpha)
        if i % (cfg.actor_update_delay + 1) == 0:  # eval freq even number
            # compensate for delay: wait X rounds, do X updates
            for _ in range(cfg.actor_update_delay):
                tlog.update(tc_update_actor(batch))
                agent.actor_updates_so_far += 1

        # update the target networks
        agent.update_targ_nets()

        if agent.timesteps_so_far % cfg.eval_every == 0:
            logger.info(("eval").upper())
            eval_start = time.time()

            for _ in range(cfg.eval_steps):
                ep = next(ep_gen)
                len_buff.append(ep["length"])
                ret_buff.append(ep["return"])

            with torch.no_grad():
                eval_metrics = {
                    "length": torch.tensor(np.array(list(len_buff)), dtype=torch.float).mean(),
                    "return": torch.tensor(np.array(list(ret_buff)), dtype=torch.float).mean(),
                }

            # log with logger
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for k, v in eval_metrics.items():
                logger.record_tabular(k, v.numpy())
            logger.dump_tabular()

            # log with wandb
            if (new_best := eval_metrics["return"].item()) > agent.best_eval_ep_ret:
                # save the new best model
                logger.info("new best eval! -- saving model to disk and wandb")
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
            wandb.log(
                {
                    **tlog.to_dict(),
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    "vitals/replay_buffer_numel": len(agent.rb),
                },
                step=agent.timesteps_so_far,
            )

            time_spent_eval += time.time() - eval_start

            if start_time is not None:
                # compute the speed in steps per second
                speed = (
                    (agent.timesteps_so_far - measure_burnin) /
                    (time.time() - start_time - time_spent_eval)
                )
                desc = f"speed={speed: 4.4f} sps"
                pbar.set_description(desc)
                wandb.log(
                    {
                        "vitals/speed": speed,
                    },
                    step=agent.timesteps_so_far,
                )

        i += 1
        tlog.clear()

    # save once we are done
    agent.save(ckpt_dir, sfx="done")
    logger.info(f"we are done -- saved model @: {ckpt_dir}")
    # mark a run as finished, and finish uploading all data (from docs)
    wandb.finish()
    logger.warn("bye")


@beartype
def evaluate(cfg: DictConfig,
             env: Env,
             agent_wrapper: Callable[[], Agent],
             name: str):

    assert isinstance(cfg, DictConfig)

    trajectory_path = None
    if cfg.gather_trajectories:
        trajectory_path = Path(cfg.trajectory_dir) / name
        trajectory_path.mkdir(parents=True, exist_ok=True)

    agent = agent_wrapper()

    agent.load(cfg.load_ckpt)

    # create episode generator
    ep_gen = episode(
        env, agent, cfg.seed, need_lists=cfg.gather_trajectories)

    pbar = tqdm.tqdm(range(cfg.num_episodes))
    pbar.set_description("evaluating")

    len_list = []
    ret_list = []

    for i in pbar:

        ep = next(ep_gen)
        len_list.append(ep_len := ep["length"])
        ret_list.append(ep_ret := ep["return"])

        if trajectory_path is not None:
            # verify that all the arrays have the same length
            for k, v in ep.items():
                assert v.shape[0] == ep_len, f"wrong array length for {k=}"
            name = f"{str(i).zfill(3)}_L{int(ep_len)}_R{int(ep_ret)}"
            td = TensorDict(ep)
            for k, v in td.items():
                if v.dtype == torch.float64:
                    # convert from float64 to float32
                    td[k] = v.float()
            fname = trajectory_path / f"{name}.h5"
            td.to_h5(fname)  # can then easily load with `from_h5`

    if trajectory_path is not None:
        logger.warn(f"saved trajectories @: {trajectory_path}")

    with torch.no_grad():
        eval_metrics = {
            "length": torch.tensor(np.array(len_list), dtype=torch.float).mean(),
            "return": torch.tensor(np.array(ret_list), dtype=torch.float).mean(),
        }

    # log with logger
    for k, v in eval_metrics.items():
        logger.record_tabular(k, v.numpy())
    logger.dump_tabular()

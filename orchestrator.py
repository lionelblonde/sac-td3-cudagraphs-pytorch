import os
import time
import h5py
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Union, Callable, ContextManager
from contextlib import contextmanager, nullcontext

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import rearrange
from termcolor import colored
import wandb
from wandb.errors import CommError
import numpy as np

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector.vector_env import VectorEnv

from helpers import logger
from helpers.opencv_util import record_video
from agents.agent import Agent


DEBUG = False


@beartype
def save_dict_h5py(save_dir: Path, name: str, data: dict[str, np.ndarray]):
    """Save dictionary containing numpy objects to h5py file."""
    fname = save_dir / Path(f"{name.zfill(3)}.h5")
    for k, v in data.items():
        assert isinstance(v, (np.ndarray, np.floating, np.integer)), f"dict['{k}']: wrong type"
    with h5py.File(fname, "w") as hf:
        for key in data:
            hf.create_dataset(key, data=data[key])


gather_roll = save_dict_h5py  # alias


@beartype
def load_dict_h5py(fname: Union[str, Path],
    ) -> tuple[dict[str, np.ndarray],
               dict[str, Union[np.floating, np.integer]]]:
    """Restore dictionary containing numpy objects from h5py file."""
    data, stts = {}, {}
    with h5py.File(fname, "r") as hf:
        for key in hf:
            dset = hf[key]
            if isinstance(dset, h5py.Dataset):
                dk = dset[()]
                assert isinstance(dk, (np.ndarray, np.floating, np.integer)), f"{type(dk) = }"
                if isinstance(dk, (np.floating, np.integer)):
                    stts[key] = dk
                else:  # last option: np.ndarray
                    data[key] = dk
            else:
                raise TypeError(f"dset for key {key} has wrong type")
    return data, stts


@beartype
def prettify_numb(n: int) -> str:
    """Display an integer number of millions, ks, etc."""
    m, k = divmod(n, 1_000_000)
    k, u = divmod(k, 1_000)
    return colored(f"{m}M {k}K {u}U", "red", attrs=["reverse"])


@beartype
@contextmanager
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
            num_env: int,
            agent: Agent,
            seed: int,
            segment_len: int,
            action_repeat: int):

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    ob, _ = env.reset(seed=seed)  # for the very first reset, we give a seed (and never again)
    ac = None  # quiets down the type-checker; as long as r is init at 0: ac will be written over

    t = 0
    r = 0  # action repeat reference

    assert agent.replay_buffers is not None

    while True:

        if r % action_repeat == 0:
            # predict action
            assert isinstance(ob, np.ndarray)
            ac = agent.predict(ob, apply_noise=True)
            # nan-proof and clip
            ac = np.nan_to_num(ac)
            ac = np.clip(ac, ac_low, ac_high)

        if t > 0 and t % segment_len == 0:
            yield

        # interact with env
        new_ob, rew, terminated, truncated, info = env.step(ac)
        rew = rearrange(rew, "b -> b 1")

        if num_env > 1:
            logger.debug(f"{terminated=} | {truncated=}")
            assert isinstance(terminated, np.ndarray)
            assert isinstance(truncated, np.ndarray)
            assert terminated.shape == truncated.shape

        if num_env == 1:
            assert isinstance(env, Env)
            done, terminated = np.array([terminated or truncated]), np.array([terminated])
            if truncated:
                logger.debug("termination caused by something like time limit or out of bounds?")
        else:
            done = np.logical_or(terminated, truncated)  # might not be used but diagnostics
            done, terminated = rearrange(done, "b -> b 1"), rearrange(terminated, "b -> b 1")
            # `done` is technically not used, but this quiets down the type-checker
        # read about what truncation means at the link below:
        # https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#truncation

        tr_or_vtr = [ob, ac, new_ob, rew, terminated]
        # note: we use terminated as a done replacement, but keep the key "dones1"

        if num_env > 1:
            pp_func = partial(postproc_vtr, num_env, info)
        else:
            assert isinstance(env, Env)
            pp_func = postproc_tr
        outs = pp_func(tr_or_vtr)
        assert outs is not None
        for i, out in enumerate(outs):  # iterate over env (although maybe only one non-vec)
            # add transition to the i-th replay buffer
            agent.replay_buffers[i].append(out)
            # log how filled the i-th replay buffer is
            logger.debug(f"rb#{i} (#entries)/capacity: {agent.replay_buffers[i].how_filled}")

        # set current state with the next
        ob = deepcopy(new_ob)

        if num_env == 1:
            assert isinstance(env, Env)
            if done:
                ob, _ = env.reset()

        t += 1
        r += 1


@beartype
def postproc_vtr(num_envs: int,
                 info: dict[str, np.ndarray],
                 vtr: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    # N.B.: for the num of envs and the workloads, serial treatment is faster than parallel
    # time it takes for the main process to spawn X threads is too much overhead
    # it starts becoming interesting if the post-processing is heavier though
    outs = []
    for i in range(num_envs):
        tr = [e[i] for e in vtr]
        if "final_observation" in info:
            if bool(info["_final_observation"][i]):
                ob, ac, _, rew, terminated = tr
                logger.debug("writing over new_ob with info[final_observation]")
                tr = [
                    ob, ac, info["final_observation"][i], rew, terminated]  # override `new_ob`
        outs.extend(postproc_tr(tr))
    return outs


@beartype
def postproc_tr(tr: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
    ob, ac, new_ob, rew, terminated = tr
    return [
        {"obs0": ob,
         "acs0": ac,
         "obs1": new_ob,
         "erews1": rew,
         "dones1": terminated}]


@beartype
def episode(env: Env,
            agent: Agent,
            seed: int):
    # generator that spits out a trajectory collected during a single episode
    # `append` operation is also significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    assert isinstance(env.action_space, gym.spaces.Box)  # to ensure `high` and `low` exist
    ac_low, ac_high = env.action_space.low, env.action_space.high

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible

    def randomize_seed() -> int:
        return seed + rng.integers(2**32 - 1, size=1).item()
        # seeded Generator: deterministic -> reproducible

    ob, _ = env.reset(seed=randomize_seed())

    cur_ep_len = 0
    cur_ep_ret = 0
    obs0 = []
    acs0 = []
    obs1 = []
    erews1 = []
    dones1 = []

    while True:

        # predict action
        ac = agent.predict(ob, apply_noise=False)
        # nan-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, ac_low, ac_high)

        obs0.append(ob)
        acs0.append(ac)
        new_ob, erew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated
        dones1.append(done)
        erews1.append(erew)
        cur_ep_len += 1
        assert isinstance(erew, float)  # quiets the type-checker
        cur_ep_ret += erew
        ob = deepcopy(new_ob)

        if done:
            obs0 = np.array(obs0)
            acs0 = np.array(acs0)
            obs1 = np.array(obs1)
            erews1 = np.array(erews1)
            dones1 = np.array(dones1)
            out = {
                "obs0": obs0,
                "acs0": acs0,
                "obs1": obs1,
                "erews1": erews1,
                "dones1": dones1,
                "ep_len": cur_ep_len,
                "ep_ret": cur_ep_ret,
            }
            yield out

            cur_ep_len = 0
            cur_ep_ret = 0
            obs0 = []
            acs0 = []
            obs1 = []
            erews1 = []
            dones1 = []

            ob, _ = env.reset(seed=randomize_seed())


@beartype
def train(cfg: DictConfig,
          env: Union[Env, VectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], Agent],
          timer_wrapper: Callable[[], Callable[[], float]],
          name: str):

    assert isinstance(cfg, DictConfig)

    # create an agent
    agent = agent_wrapper()

    # create a timer
    timer = timer_wrapper()

    # create context manager
    @beartype
    def ctx(op: str) -> ContextManager:
        return timed(op, timer) if DEBUG else nullcontext()

    # set up model save directory
    ckpt_dir = Path(cfg.checkpoint_dir) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vid_dir = Path(cfg.video_dir) / name
    if cfg.record:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # save the model as a dry run, to avoid bad surprises at the end
    agent.save(ckpt_dir, sfx="dryrun")
    logger.warn(f"dry run -- saved model @:\n{ckpt_dir}")

    # group by everything except the seed, which is last, hence index -1
    # it groups by uuid + gitSHA + env_id
    group = ".".join((ename := name).split(".")[:-1])  # nitpicking walrus for alignment
    logger.warn(f"{ename=}")
    logger.warn(f"{group=}")
    # set up wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
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

    for glob in ["train_actr", "train_crit", "eval"]:  # wandb categories
        # define a custom x-axis
        wandb.define_metric(f"{glob}/step")
        wandb.define_metric(f"{glob}/*", step_metric=f"{glob}/step")

    # create segment generator for training the agent
    roll_gen = segment(
        env, cfg.num_env, agent, cfg.seed, cfg.segment_len, cfg.action_repeat)
    # create episode generator for evaluating the agent
    eval_seed = cfg.seed + 123456  # arbitrary choice
    ep_gen = episode(eval_env, agent, eval_seed)

    i = 0

    while agent.timesteps_so_far <= cfg.num_timesteps:

        logger.info((f"iter#{i}").upper())
        if i % cfg.eval_every == 0:
            logger.warn((f"iter#{i}").upper())
            # so that when logger level is WARN, we see the iter number before the the eval metrics

        logger.info(("interact").upper())
        its = timer()
        next(roll_gen)  # no need to get the returned segment, stored in buffer
        agent.timesteps_so_far += cfg.segment_len
        logger.info(f"so far {prettify_numb(agent.timesteps_so_far)} steps made")
        logger.info(colored(
            f"interaction time: {timer() - its}secs",
            "green"))

        logger.info(("train").upper())

        tts = timer()
        ttl = []
        for _ in range(tot := cfg.training_steps_per_iter):

            # sample a batch of transitions and trajectories
            trns_batch = agent.sample_trns_batch()
            # determine if updating the actr
            update_actr = not bool(agent.crit_updates_so_far % cfg.actor_update_delay)
            with ctx("actor-critic training"):
                # update the actor and critic
                agent.update_actr_crit(trns_batch, update_actr=update_actr)

            ttl.append(timer() - tts)
            tts = timer()

        logger.info(colored(
            f"avg tt over {tot}steps: {(avg_tt_per_iter := np.mean(ttl))}secs",  # logged in eval
            "green", attrs=["reverse"]))
        logger.info(colored(
            f"tot tt over {tot}steps: {np.sum(ttl)}secs",
            "magenta", attrs=["reverse"]))

        i += 1

        if i % cfg.eval_every == 0:

            logger.info(("eval").upper())

            len_buff, ret_buff = [], []

            for _ in range(cfg.eval_steps_per_iter):

                # sample an episode
                ep = next(ep_gen)

                len_buff.append(ep["ep_len"])
                ret_buff.append(ep["ep_ret"])

            eval_metrics: dict[str, np.floating] = {  # type-checker
                "ep_len-mean": np.mean(np.array(len_buff)),
                "ep_ret-mean": np.mean(np.array(ret_buff))}

            if (new_best := eval_metrics["ep_ret-mean"].item()) > agent.best_eval_ep_ret:
                # save the new best model
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
                logger.warn(f"new best eval! -- saved model @:\n{ckpt_dir}")

            # log stats in csv
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for kv in eval_metrics.items():
                logger.record_tabular(*kv)
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            # log stats in dashboard
            assert agent.replay_buffers is not None
            wandb_dict = {
                **eval_metrics,
                "rbx-num-entries": np.array(agent.replay_buffers[0].num_entries),
                # taking the first because this one will always exist whatever the numenv
                "avg-tt-per-iter": avg_tt_per_iter}
            agent.send_to_dash(
                wandb_dict,
                step_metric=agent.timesteps_so_far,
                glob="eval",
            )

        logger.info()

    # save once we are done
    agent.save(ckpt_dir, sfx="done")
    logger.warn(f"we are done -- saved model @:\n{ckpt_dir}\nbye.")
    # mark a run as finished, and finish uploading all data (from docs)
    wandb.finish()

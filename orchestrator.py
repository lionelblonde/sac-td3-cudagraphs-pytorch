import os
import time
import tqdm
from pathlib import Path
from functools import partial
from typing import Union, Callable
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

from gymnasium.core import Env
from gymnasium.vector.vector_env import VectorEnv

from helpers import logger
from agents.agent import Agent


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
            device: torch.device,
            seed: int,
            segment_len: int,
            learning_starts: int,
            action_repeat: int):

    ob, _ = env.reset(seed=seed)  # for the very first reset, we give a seed (and never again)
    ob = torch.as_tensor(ob, device=device, dtype=torch.float)
    ac = None  # quiets down the type-checker; as long as r is init at 0: ac will be written over

    t = 0
    r = 0  # action repeat reference

    assert agent.replay_buffers is not None

    while True:

        if r % action_repeat == 0:
            # predict action
            if agent.timesteps_so_far < learning_starts:
                ac = env.action_space.sample()
            else:
                assert isinstance(ob, torch.Tensor)
                ac = agent.predict(ob, explore=True)

        if t > 0 and t % segment_len == 0:
            yield

        # interact with env
        new_ob, rew, terminated, truncated, infos = env.step(ac)

        ac = torch.as_tensor(ac, device=device, dtype=torch.float)

        new_ob = torch.as_tensor(new_ob, device=device, dtype=torch.float)

        if num_env == 1:
            rew = np.array([rew])
        rew = rearrange(rew, "b -> b 1")
        rew = torch.as_tensor(rew, device=device, dtype=torch.float)

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
            pp_func = partial(postproc_vtr, num_env, device, infos)
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
        ob = new_ob

        if num_env == 1:
            assert isinstance(env, Env)
            if done:
                ob, _ = env.reset()

        t += 1
        r += 1


@beartype
def postproc_vtr(num_envs: int,
                 device: torch.device,
                 infos: dict[str, np.ndarray],
                 vtr: list[Union[np.ndarray, torch.Tensor]],
    ) -> list[dict[str, Union[np.ndarray, torch.Tensor]]]:
    # N.B.: for the num of envs and the workloads, serial treatment is faster than parallel
    # time it takes for the main process to spawn X threads is too much overhead
    # it starts becoming interesting if the post-processing is heavier though
    outs = []
    for i in range(num_envs):
        tr = [e[i] for e in vtr]
        if "final_observation" in infos:
            if bool(infos["_final_observation"][i]):
                ob, ac, _, rew, terminated = tr
                logger.debug("writing over new_ob with info[final_observation]")
                real_new_ob = torch.as_tensor(
                    infos["final_observation"][i], device=device, dtype=torch.float)
                tr = [ob, ac, real_new_ob, rew, terminated]  # override `new_ob`
        outs.extend(postproc_tr(tr))
    return outs


@beartype
def postproc_tr(tr: list[Union[np.ndarray, torch.Tensor]],
    ) -> list[dict[str, Union[np.ndarray, torch.Tensor]]]:
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
            device: torch.device,
            seed: int):
    # generator that spits out a trajectory collected during a single episode
    # `append` operation is also significantly faster on lists than numpy arrays,
    # they will be converted to numpy arrays once complete right before the yield

    rng = np.random.default_rng(seed)  # aligned on seed, so always reproducible

    def randomize_seed() -> int:
        return seed + rng.integers(2**32 - 1, size=1).item()
        # seeded Generator: deterministic -> reproducible

    obs0 = []
    acs0 = []
    obs1 = []
    erews1 = []
    dones1 = []
    ep_len = 0
    ep_ret = 0

    ob, _ = env.reset(seed=randomize_seed())
    obs0.append(ob)
    ob = torch.as_tensor(ob, device=device, dtype=torch.float)

    while True:

        # predict action
        assert isinstance(ob, torch.Tensor)
        ac = agent.predict(ob, explore=False)

        acs0.append(ac)
        new_ob, erew, terminated, truncated, infos = env.step(ac)

        done = terminated or truncated
        dones1.append(done)
        erews1.append(erew)

        obs1.append(new_ob)
        if not done:
            obs0.append(new_ob)

        new_ob = torch.as_tensor(new_ob, device=device, dtype=torch.float)
        ob = new_ob

        if "final_info" in infos:
            assert len(infos["final_info"]) == 1
            for info in infos["final_info"]:
                ep_len = float(info["episode"]["l"])
                ep_ret = float(info["episode"]["r"])

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
                "ep_len": ep_len,
                "ep_ret": ep_ret,
            }
            yield out

            obs0 = []
            acs0 = []
            obs1 = []
            erews1 = []
            dones1 = []

            ob, _ = env.reset(seed=randomize_seed())
            obs0.append(ob)
            ob = torch.as_tensor(ob, device=device, dtype=torch.float)


@beartype
def train(cfg: DictConfig,
          env: Union[Env, VectorEnv],
          eval_env: Env,
          agent_wrapper: Callable[[], Agent],
          name: str,
          device: torch.device):

    assert isinstance(cfg, DictConfig)

    agent = agent_wrapper()

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
                save_code=True,
            )
            break
        except CommError:
            pause = 10
            logger.info(f"wandb co error. Retrying in {pause} secs.")
            time.sleep(pause)
    logger.info("wandb co established!")

    # create segment generator for training the agent
    roll_gen = segment(
        env,
        cfg.num_env,
        agent,
        device,
        cfg.seed,
        cfg.segment_len,
        cfg.learning_starts,
        cfg.action_repeat,
    )
    # create episode generator for evaluating the agent
    eval_seed = cfg.seed + 123456  # arbitrary choice
    ep_gen = episode(
        eval_env,
        agent,
        device,
        eval_seed,
    )

    i = 0
    start_time = None
    measure_burnin = None
    pbar = tqdm.tqdm(range(cfg.num_timesteps))
    time_spent_eval = 0

    tlog = {}
    elog = {}
    maxlen = 20 * cfg.eval_steps
    len_buff = deque(maxlen=maxlen)
    ret_buff = deque(maxlen=maxlen)

    while agent.timesteps_so_far <= cfg.num_timesteps:

        if ((agent.timesteps_so_far >= (cfg.measure_burnin + cfg.learning_starts) and
             start_time is None)):
            start_time = time.time()
            measure_burnin = agent.timesteps_so_far

        logger.info(("interact").upper())
        next(roll_gen)
        agent.timesteps_so_far += (increment := cfg.segment_len * cfg.num_env)
        pbar.update(increment)
        logger.info(f"so far {prettify_numb(agent.timesteps_so_far)} steps made")

        if agent.timesteps_so_far <= cfg.learning_starts:
            # start training when enough data
            pbar.set_description("not learning yet")
            i += 1
            continue

        logger.info(("train").upper())
        for _ in range(cfg.training_steps_per_iter):
            # sample a batch of transitions
            trns_batch = agent.sample_trns_batch()
            # assemble the loss operands
            operands = agent.build_loss_operands(trns_batch)
            # compute the losses
            actor_loss, qf_loss, loga_loss = agent.compute_losses(*operands)
            # update the online networks
            if not cfg.actor_update_delay or bool(agent.qnet_updates_so_far % 2):
                agent.update_actor(actor_loss, loga_loss)
                agent.actor_updates_so_far += 1
                tlog.update(
                    {
                        "loss/actor": actor_loss,
                    },
                )
                if loga_loss is not None:
                    tlog.update(
                        {
                            "loss/loga": loga_loss,
                            "vitals/alpha": agent.alpha,
                        },
                    )
            agent.update_crit(qf_loss)
            agent.qnet_updates_so_far += 1
            tlog.update(
                {
                    "loss/q": qf_loss,
                },
            )
            # update the target networks
            agent.update_targ_nets()

        if (agent.timesteps_so_far % cfg.eval_every == 0):
            logger.info(("eval").upper())
            eval_start = time.time()

            for _ in range(cfg.eval_steps):
                ep = next(ep_gen)
                len_buff.append(ep["ep_len"])
                ret_buff.append(ep["ep_ret"])

            with torch.no_grad():
                eval_metrics = {
                    "length": torch.tensor(list(len_buff), dtype=torch.float).mean(),
                    "return": torch.tensor(list(ret_buff), dtype=torch.float).mean(),
                }

            if (new_best := eval_metrics["return"].item()) > agent.best_eval_ep_ret:
                # save the new best model
                agent.best_eval_ep_ret = new_best
                agent.save(ckpt_dir, sfx="best")
                logger.info(f"new best eval! -- saved model @: {ckpt_dir}")

            # log with logger
            logger.record_tabular("timestep", agent.timesteps_so_far)
            for k, v in eval_metrics.items():
                logger.record_tabular(k, v.numpy())
            logger.dump_tabular()

            # log with wandb
            assert agent.replay_buffers is not None
            elog = {
                **{f"eval/{k}": v for k, v in eval_metrics.items()},
            }
            wandb.log(
                {
                    **tlog,
                    **elog,
                },
                step=agent.timesteps_so_far,
            )

            time_spent_eval += time.time() - eval_start

            if start_time is not None:
                assert measure_burnin is not None
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
        tlog = {}
        elog = {}

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
             name: str,
             device: torch.device):

    assert isinstance(cfg, DictConfig)

    trajectory_path = None
    if cfg.gather_trajectories:
        trajectory_path = Path(cfg.trajectory_dir) / name
        trajectory_path.mkdir(parents=True, exist_ok=True)

    agent = agent_wrapper()

    agent.load(cfg.load_ckpt)

    # create episode generator
    ep_gen = episode(env, agent, device, cfg.seed)

    pbar = tqdm.tqdm(range(cfg.num_episodes))
    pbar.set_description("evaluating")

    len_buff = []
    ret_buff = []

    for i in pbar:

        ep = next(ep_gen)
        len_buff.append(ep_len := ep["ep_len"])
        ret_buff.append(ep_ret := ep["ep_ret"])

        if trajectory_path is not None:
            name = f"{str(i).zfill(3)}_L{ep_len}_R{ep_ret}"
            td = TensorDict(ep)
            fname = trajectory_path / f"{name}.h5"
            td.to_h5(fname)  # can then easily load with `from_h5`

    if trajectory_path is not None:
        logger.warn(f"saved trajectories @: {trajectory_path}")

    with torch.no_grad():
        eval_metrics = {
            "length": torch.tensor(list(len_buff), dtype=torch.float).mean(),
            "return": torch.tensor(list(ret_buff), dtype=torch.float).mean(),
        }

    # log with logger
    for k, v in eval_metrics.items():
        logger.record_tabular(k, v.numpy())
    logger.dump_tabular()

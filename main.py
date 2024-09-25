import os
import time
import subprocess
from pathlib import Path
from typing import Optional, Callable

from beartype import beartype
import fire
from omegaconf import OmegaConf, DictConfig
import random
import numpy as np
import torch

from gymnasium.core import Env

import orchestrator
from helpers import logger
from helpers.env_makers import make_env
from agents.memory import ReplayBuffer
from agents.agent import Agent


@beartype
def make_uuid(num_syllables: int = 2, num_parts: int = 3) -> str:
    """Randomly create a semi-pronounceable uuid"""
    part1 = ["s", "t", "r", "ch", "b", "c", "w", "z", "h", "k", "p", "ph", "sh", "f", "fr"]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ "-", "_", "."]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for _ in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for _ in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


@beartype
def get_name(uuid: str, env_id: str, seed: int) -> str:
    """Assemble long experiment name"""
    name = uuid
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        sha = out.strip().decode("ascii")
        name += f".gitSHA_{sha}"
    except OSError:
        pass
    name += f".{env_id}"
    name += f".seed{str(seed).zfill(2)}"
    return name


class MagicRunner(object):

    DISABLE_LOGGER: bool = False
    LOGGER_LEVEL: int = logger.WARN
    PRODUCTION_MODE: bool = True

    @beartype
    def __init__(self, cfg: str,  # give the relative path to cfg here
                 env_id: str,  # never in cfg: always give one in arg
                 seed: int,  # never in cfg: always give one in arg
                 wandb_project: Optional[str] = None,  # is either given in arg (prio) or in cfg
                 uuid: Optional[str] = None,  # never in cfg, but not forced to give in arg either
                 load_ckpt: Optional[str] = None):  # same as uuid: from arg or nothing

        logger.configure_default_logger()

        # retrieve config from filesystem
        proj_root = Path(__file__).resolve().parent
        _cfg = OmegaConf.load(proj_root / Path(cfg))
        assert isinstance(_cfg, DictConfig)
        self._cfg: DictConfig = _cfg  # for the type-checker

        logger.info("the config loaded:")
        logger.info(OmegaConf.to_yaml(self._cfg))

        self._cfg.root = str(proj_root)  # in config: used by wandb
        for k in ("checkpoints", "logs", "videos", "rolls"):
            new_k = f"{k[:-1]}_dir"
            self._cfg[new_k] = str(proj_root / k)  # for yml saving

        # set only if nonexistant key in cfg
        self._cfg.seed = seed
        self._cfg.env_id = env_id

        assert "wandb_project" in self._cfg  # if not in cfg from fs, abort
        if wandb_project is not None:
            self._cfg.wandb_project = wandb_project  # overwrite cfg

        assert "uuid" not in self._cfg  # uuid should never be in the cfg file
        self._cfg.uuid = uuid if uuid is not None else make_uuid()

        assert "load_ckpt" not in self._cfg, "load_ckpt must never be in the cfg file"
        if load_ckpt is not None:
            self._cfg.load_ckpt = load_ckpt  # add in cfg
        else:
            logger.info("no ckpt to load: key will not exist in cfg")

        self.name = get_name(self._cfg.uuid, self._cfg.env_id, self._cfg.seed)

        # slight overwrite for consistency, before setting to read-only
        self._cfg.num_env = self._cfg.numenv if self._cfg.vectorized else 1

        # set the cfg to read-only for safety
        OmegaConf.set_readonly(self._cfg, value=True)

    @beartype
    def setup_device(self) -> torch.device:
        assert not self._cfg.fp16 or self._cfg.cuda, "fp16 => cuda"  # TODO(lionel): fp16 support
        if self._cfg.cuda:
            # use cuda
            assert torch.cuda.is_available()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda:0")
        else:
            # default case: just use plain old cpu, no cuda or m-chip gpu
            device = torch.device("cpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # kill any possibility of usage
        logger.info(f"device in use: {device}")
        return device

    @beartype
    def train(self):

        # mlsys
        torch.set_num_threads(self._cfg.num_env)
        # TODO(lionel): keep an eye on this

        # set printing options
        np.set_printoptions(precision=3)

        # logger
        if self.DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            log_path = Path(self._cfg.log_dir) / self.name
            log_path.mkdir(parents=True, exist_ok=True)
            logger.configure(directory=log_path, format_strs=["stdout", "log", "json", "csv"])
            logger.set_level(self.LOGGER_LEVEL)
            # config dump
            OmegaConf.save(config=self._cfg, f=(log_path / "cfg.yml"))

        # device
        device = self.setup_device()

        # seed
        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed_all(self._cfg.seed)

        # env
        env, net_shapes, erb_shapes, max_ac = make_env(
            self._cfg.env_id,
            self._cfg.horizon,
            vectorized=self._cfg.vectorized,
            multi_proc=self._cfg.multi_proc,
            num_env=self._cfg.num_env,
            record=False,
            render=self._cfg.render,
        )

        # create an agent wrapper

        replay_buffers = [ReplayBuffer(
            generator=torch.Generator(device).manual_seed(self._cfg.seed),
            capacity=self._cfg.rbx_capacity,
            erb_shapes=erb_shapes,
            device=device,
        ) for _ in range(self._cfg.num_env)]
        for i, rb in enumerate(replay_buffers):
            logger.info(f"rb#{i} [{rb}] is set")

        # perform quick sanity check on a ring buffer data structure
        replay_buffers[0].ring_buffers["acs0"].sanity_check_ringbuffer()

        @beartype
        def agent_wrapper() -> Agent:
            return Agent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                actr_noise_rng=torch.Generator(device).manual_seed(self._cfg.seed),
                replay_buffers=replay_buffers,
            )

        @beartype
        def timer_wrapper() -> Callable[[], float]:
            def _timer() -> float:
                if self._cfg.cuda and not self.PRODUCTION_MODE:
                    logger.debug("cuda syncing clocks")
                    torch.cuda.synchronize()
                return time.time()
            return _timer

        # create an evaluation environment not to mess up with training rollouts
        eval_env, _, _, _ = make_env(
            self._cfg.env_id,
            horizon=1000,
            vectorized=False,
            multi_proc=False,
            record=self._cfg.record,
            render=self._cfg.render,
        )
        assert isinstance(eval_env, Env), "no vecenv allowed here"

        # train
        orchestrator.learn(
            cfg=self._cfg,
            env=env,
            eval_env=eval_env,
            agent_wrapper=agent_wrapper,
            timer_wrapper=timer_wrapper,
            name=self.name,
        )

        # cleanup
        env.close()
        eval_env.close()

    @beartype
    def evaluate(self):

        # mlsys
        torch.set_num_threads(1)

        # set printing options
        np.set_printoptions(precision=3)

        # logger
        if self.DISABLE_LOGGER:
            logger.set_level(logger.DISABLED)  # turn the logging off
        else:
            logger.configure(directory=None, format_strs=["stdout"])
            logger.set_level(self.LOGGER_LEVEL)

        # device
        device = self.setup_device()

        # seed
        torch.manual_seed(self._cfg.seed)
        torch.cuda.manual_seed_all(self._cfg.seed)

        # env
        env, net_shapes, _, max_ac = make_env(
            self._cfg.env_id,
            horizon=1000,
            vectorized=False,
            multi_proc=False,
            record=self._cfg.record,
            render=self._cfg.render,
        )
        assert isinstance(env, Env), "no vecenv allowed here"

        @beartype
        def agent_wrapper() -> Agent:
            return Agent(
                net_shapes=net_shapes,
                max_ac=max_ac,
                device=device,
                hps=self._cfg,
                actr_noise_rng=torch.Generator(device).manual_seed(self._cfg.seed),
                replay_buffers=None,
            )

        # evaluate
        orchestrator.evaluate(
            cfg=self._cfg,
            env=env,
            agent_wrapper=agent_wrapper,
            name=self.name,
        )

        # cleanup
        env.close()


if __name__ == "__main__":
    fire.Fire(MagicRunner)

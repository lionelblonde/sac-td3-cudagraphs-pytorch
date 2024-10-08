import itertools
from copy import deepcopy
import os
import sys
import shutil
import socket

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
import fire
import numpy as np
import subprocess
import yaml
from pathlib import Path
from typing import Any

from helpers import logger
from main import make_uuid


ENV_BUNDLES: dict[str, list[str]] = {
    "debug": [
        "Hopper-v4",
    ],
    "low": [
        "Hopper-v4",
        "Pusher-v4",
    ],
    "medium": [
        "HalfCheetah-v4",
        "Walker2d-v4",
        "Ant-v4",
    ],
    "high": [
        "Humanoid-v4",
        "HumanoidStandup-v4",
    ],
}
# for each environment bundle, set the GPU mem to request
GPU_MEM_MAP: dict[str, int] = {
    "debug": 10,
    "low": 10,
    "medium": 10,
    "high": 20,
}
assert set(ENV_BUNDLES.keys()) == set(GPU_MEM_MAP.keys())

GPU_MEMORY = 20
MEMORY = 16
NUM_SWEEP_TRIALS = 10
DEST_DIR_AUTOGEN_CFG = Path("tasks/autogen")


class Spawner(object):

    @beartype
    def __init__(self,
                 cfg: str,
                 num_seeds: int,
                 env_bundle: str,
                 caliber: str,
                 deployment: str,
                 *,
                 sweep: bool):

        self.num_seeds = num_seeds
        self.deployment = deployment
        self.sweep = sweep

        assert self.deployment in {"tmux", "slurm"}

        self.uuid = make_uuid()

        # retrieve config from filesystem
        self.proj_root = Path(__file__).resolve().parent
        old_path_to_cfg = self.proj_root / Path(cfg)
        _cfg = OmegaConf.load(old_path_to_cfg)
        assert isinstance(_cfg, DictConfig)
        self._cfg = _cfg

        new_path_to_cfg_dir = self.proj_root / DEST_DIR_AUTOGEN_CFG / self.uuid
        new_path_to_cfg_dir.mkdir(parents=True, exist_ok=True)
        new_path_to_cfg = new_path_to_cfg_dir / old_path_to_cfg.name
        shutil.copy(old_path_to_cfg, new_path_to_cfg)
        self.path_to_cfg = new_path_to_cfg

        logger.info("the config loaded:")
        logger.info(OmegaConf.to_yaml(self._cfg))

        # assemble wandb project name
        self.wandb_project = f"{self._cfg.wandb_project}-{self.deployment}"
        # define spawn type
        self.job_type = "sweep" if self.sweep else "fixed"

        if self.deployment == "slurm":
            # translate intuitive caliber into duration and cluster partition
            calibers = {
                "veryshort": "0-01:00:00",
                "short": "0-02:00:00",
                "ok": "0-04:00:00",
                "long": "0-06:00:00",
                "verylong": "0-12:00:00",
                "veryverylong": "1-00:00:00",
                "veryveryverylong": "2-00:00:00",
                "veryveryveryverylong": "4-00:00:00",
            }
            self.duration = calibers[caliber]  # KeyError trigger if invalid caliber

            hostname = socket.gethostname()
            if "verylong" in caliber:
                if self._cfg.cuda:
                    if "yggdrasil" in hostname:
                        self.partition = "public-gpu"
                    else:
                        self.partition = "private-cui-gpu,private-kalousis-gpu"
                elif "yggdrasil" in hostname:
                    self.partition = "public-cpu,private-cui-cpu,public-longrun-cpu"
                else:
                    self.partition = "public-cpu,private-cui-cpu"
            elif self._cfg.cuda:  # < verylong and gpu
                if "yggdrasil" in hostname:
                    self.partition = "shared-gpu"
                else:
                    self.partition = "shared-gpu,private-cui-gpu,private-kalousis-gpu"
            else:   # < verylong and cpu
                self.partition = "shared-cpu,public-cpu,private-cui-cpu"

        # define the set of considered environments from the considered suite
        self.envs = ENV_BUNDLES[env_bundle]
        # also use the bundle name to determine the GPU memory to request
        self.gpu_memory = GPU_MEM_MAP[env_bundle]

    @staticmethod
    @beartype
    def copy_and_add_seed(hpmap: dict[str, Any], seed: int) -> dict[str, Any]:
        hpmap_ = deepcopy(hpmap)
        hpmap_.update({"seed": seed})
        return hpmap_

    @staticmethod
    @beartype
    def copy_and_add_env(hpmap: dict[str, Any], env: str) -> dict[str, Any]:
        hpmap_ = deepcopy(hpmap)
        hpmap_.update({"env_id": env})
        return hpmap_

    @beartype
    def get_hps(self) -> list[dict[str, Any]]:
        """Return a list of maps of hyperparameters"""

        # assemble the hyperparameter map
        hpmap = {
            "cfg": self.path_to_cfg,
            "wandb_project": self.wandb_project,
            "uuid": self.uuid,
        }

        if self.sweep:
            # random search: replace some entries with random values
            rng = np.random.default_rng(seed=654321)
            hpmap.update({
                "batch_size": int(rng.choice([128, 256, 512])),
            })

        # carry out various duplications

        # duplicate for each environment
        hpmaps = [self.copy_and_add_env(hpmap, env) for env in self.envs]

        # duplicate for each seed
        hpmaps = [self.copy_and_add_seed(hpmap_, seed)
                  for hpmap_ in hpmaps
                  for seed in range(self.num_seeds)]

        # verify that the correct number of configs have been created
        assert len(hpmaps) == self.num_seeds * len(self.envs)

        return hpmaps

    @staticmethod
    @beartype
    def unroll_options(hpmap: dict[str, Any]) -> str:
        """Transform the dictionary of hyperparameters into a string of bash options"""
        arguments = ""
        for k, v in hpmap.items():
            arguments += f" --{k}={v}"
        return arguments

    @beartype
    def create_job_str(self, name: str, command: str) -> str:
        """Build the batch script that launches a job"""

        # prepend python command with python binary path
        cmd = Path(os.environ["CONDA_PREFIX"]) / "bin" / command

        if self.deployment == "slurm":
            Path("./out").mkdir(exist_ok=True)
            # set sbatch cfg
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"#SBATCH --job-name={name}\n"
                                f"#SBATCH --partition={self.partition}\n"
                                f"#SBATCH --nodes=1\n"
                                f"#SBATCH --ntasks=1\n"
                                f"#SBATCH --cpus-per-task={self._cfg.num_envs}\n"
                                f"#SBATCH --time={self.duration}\n"
                                f"#SBATCH --mem={MEMORY}000\n"
                                "#SBATCH --output=./out/run_%j.out\n")
            if self.deployment == "slurm":
                # Sometimes versions are needed (some clusters)
                if self._cfg.cuda:
                    bash_script_str += (f"#SBATCH --gres=gpu:1,VramPerGpu:{self.gpu_memory}G\n")
                bash_script_str += ("\n")

            # load modules
            bash_script_str += ("module load GCC/9.3.0\n")
            if self._cfg.cuda:
                bash_script_str += ("module load CUDA/11.5.0\n")

            # sometimes!? bash_script_str += ("module load Mesa/19.2.1\n")

            bash_script_str += ("\n")

            # launch command
            if self.deployment == "slurm":
                bash_script_str += (f"srun {cmd}")

        elif self.deployment == "tmux":
            # set header
            bash_script_str = ("#!/usr/bin/env bash\n\n")
            bash_script_str += (f"# job name: {name}\n\n")
            # launch command
            bash_script_str += (f"{cmd}")  # left in this format for easy edits

        else:
            raise NotImplementedError("cluster selected is not covered.")

        return bash_script_str


@beartype
def run(cfg: str,
        conda_env: str,
        env_bundle: str,
        deployment: str,
        num_seeds: int,
        caliber: str,
        *,
        deploy_now: bool,
        sweep: bool = False,
        wandb_upgrade: bool = False,
        wandb_dryrun: bool = False,
        debug: bool = False):
    """Spawn jobs"""

    logger.configure_default_logger()
    logger.set_level(logger.INFO)

    if wandb_upgrade:
        # upgrade the wandb package
        logger.info("::::upgrading wandb pip package")
        out = subprocess.check_output([
            sys.executable, "-m", "pip", "install", "wandb", "--upgrade",
        ])
        logger.info(out.decode("utf-8"))

    if wandb_dryrun:
        # run wandb in offline mode (does not sync with wandb servers in real time,
        # use `wandb sync` later on the local directory in `wandb/`
        # to sync to the wandb cloud hosted app)
        os.environ["WANDB_MODE"] = "dryrun"

    # create a spawner object
    spawner = Spawner(cfg, num_seeds, env_bundle, caliber, deployment, sweep=sweep)

    # create directory for spawned jobs
    spawn_dir = Path(spawner.proj_root) / "spawn"
    spawn_dir.mkdir(exist_ok=True)
    tmux_dir = spawner.proj_root / "tmux"  # create name to prevent unbound from type-checker
    if deployment == "tmux":
        Path(tmux_dir).mkdir(exist_ok=True)

    # get the hyperparameter set(s)
    if sweep:
        hpmaps_ = [spawner.get_hps() for _ in range(NUM_SWEEP_TRIALS)]
        # flatten into a 1-dim list
        hpmaps = [x for hpmap in hpmaps_ for x in hpmap]
    else:
        hpmaps = spawner.get_hps()

    # create associated task strings
    commands = [f"python -O main.py train{spawner.unroll_options(hpmap)}" for hpmap in hpmaps]
    # N.B.: the `-O` option is very important: beartype sees __debug__ is turns itself off
    if not len(commands) == len(set(commands)):
        # terminate in case of duplicate experiment (extremely unlikely though)
        raise ValueError("bad luck, there are dupes -> try again (:")
    # create the job maps
    names = [f"{spawner.job_type}.{hpmap['uuid']}_{i}" for i, hpmap in enumerate(hpmaps)]

    # finally get all the required job strings
    jobs = itertools.starmap(spawner.create_job_str, zip(names, commands))

    # spawn the jobs
    for i, (name, job) in enumerate(zip(names, jobs)):
        logger.info(f"job#{i},name={name} -> ready to be deployed.")
        if debug:
            logger.info("cfg below.")
            logger.info(job + "\n")
        dirname = name.split(".")[1]
        full_dirname = Path(spawn_dir) / dirname
        full_dirname.mkdir(exist_ok=True)
        job_name = full_dirname / f"{name}.sh"
        job_name.write_text(job)
        if deploy_now and deployment != "tmux":
            # spawn the job!
            stdout = subprocess.run(["sbatch", job_name], check=True).stdout
            if debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(f"job#{i},name={name} -> deployed on slurm.")

    if deployment == "tmux":
        dir_ = hpmaps[0]["uuid"].split(".")[0]  # arbitrarilly picked index 0
        session_name = f"{spawner.job_type}-{str(num_seeds).zfill(2)}seeds-{dir_}"
        yaml_content = {"session_name": session_name,
                        "windows": [],
                        "environment": {}}
        for i, name in enumerate(names):
            executable = f"{name}.sh"
            pane = {"shell_command": [f"source activate {conda_env}",
                                      f"chmod u+x spawn/{dir_}/{executable}",
                                      f"spawn/{dir_}/{executable}"]}
            window = {"window_name": f"job{str(i).zfill(2)}",
                      "focus": False,
                      "panes": [pane]}
            yaml_content["windows"].append(window)
            logger.info(
                f"job#{i},name={name} -> will run in tmux, session={session_name},window={i}.",
            )

        # dump the assembled tmux cfg into a yaml file
        job_config = Path(tmux_dir) / f"{session_name}.yaml"
        job_config.write_text(yaml.dump(yaml_content, default_flow_style=False))
        if deploy_now:
            # spawn all the jobs in the tmux session!
            stdout = subprocess.run(["tmuxp", "load", "-d", job_config], check=True).stdout
            if debug:
                logger.info(f"[STDOUT]\n{stdout}")
            logger.info(
                f"[{len(list(jobs))}] jobs are now running in tmux session =={session_name}==.",
            )


if __name__ == "__main__":
    fire.Fire(run)

import tempfile
from pathlib import Path
from typing import Optional, Union
from collections import defaultdict

from beartype import beartype
from omegaconf import DictConfig
from einops import pack
import wandb
import numpy as np
import torch
import torch.special
from torch.optim import Adam
from torch.nn.utils import clip_grad as cg
from torch.nn import functional as ff

from helpers import logger
from helpers.normalizer import RunningMoments
from agents.nets import log_module_info, Actor, Critic
from agents.memory import ReplayBuffer


class NormalActionNoise(object):

    @beartype
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, generator: torch.Generator):
        """Additive action space Gaussian noise"""
        assert isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor)
        self.mu = mu
        self.sigma = sigma
        self.device = self.mu.device  # grab the device we are on (assumed sigma and mu on same)
        self.rng = generator

    @beartype
    def generate(self):
        return torch.normal(self.mu, self.sigma, generator=self.rng).to(self.device)

    @beartype
    def __repr__(self):
        return f"NormalAcNoise(mu={self.mu}, sigma={self.sigma})"


class Agent(object):

    TRAIN_METRICS_WANDB_LOG_FREQ: int = 100

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 max_ac: float,
                 device: torch.device,
                 hps: DictConfig,
                 actr_noise_rng: torch.Generator,
                 replay_buffers: Optional[list[ReplayBuffer]]):
        self.ob_shape, self.ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]
        # the self here needed because those shapes are used in the orchestrator
        self.max_ac = max_ac

        self.device = device

        assert isinstance(hps, DictConfig)
        self.hps = hps

        self.timesteps_so_far = 0
        self.actr_updates_so_far = 0
        self.crit_updates_so_far = 0

        self.best_eval_ep_ret = -np.inf  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay buffer
        self.replay_buffers = replay_buffers

        # setup action noise
        self.ac_noise = NormalActionNoise(
            mu=torch.zeros(self.ac_shape).to(self.device),
            sigma=float(self.hps.normal_noise_std) * torch.ones(self.ac_shape).to(self.device),
            generator=actr_noise_rng,
        )  # spherical/isotropic additive Normal(0., 0.1) action noise (we set the std via cfg)
        logger.debug(f"{self.ac_noise} configured")

        # create observation normalizer that maintains running statistics
        self.rms_obs = RunningMoments(shape=self.ob_shape, device=self.device)

        if self.hps.ret_norm:
            # create return normalizer that maintains running statistics
            self.rms_ret = RunningMoments(shape=(1,), device=self.device)

        # create online and target nets

        actr_hid_dims = (300, 200)
        crit_hid_dims = (400, 300)

        actr_net_args = [self.ob_shape, self.ac_shape, actr_hid_dims, self.rms_obs, self.max_ac]
        actr_net_kwargs = {"layer_norm": self.hps.layer_norm}
        self.actr = Actor(*actr_net_args, **actr_net_kwargs).to(self.device)
        self.targ_actr = Actor(*actr_net_args, **actr_net_kwargs).to(self.device)

        crit_net_args = [self.ob_shape, self.ac_shape, crit_hid_dims, self.rms_obs]
        crit_net_kwargs = {"layer_norm": self.hps.layer_norm}
        self.crit = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
        self.targ_crit = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)

        # initilize the target nets
        self.targ_actr.load_state_dict(self.actr.state_dict())
        self.targ_crit.load_state_dict(self.crit.state_dict())

        if self.hps.clipped_double:
            # create second ("twin") critic and target critic
            # ref: TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
            self.targ_twin = Critic(*crit_net_args, **crit_net_kwargs).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        # set up the optimizers

        self.actr_opt = Adam(self.actr.parameters(), lr=self.hps.actr_lr)
        self.crit_opt = Adam(self.crit.parameters(), lr=self.hps.crit_lr)
        if self.hps.clipped_double:
            self.twin_opt = Adam(self.twin.parameters(), lr=self.hps.crit_lr)

        # log module architectures
        log_module_info(self.actr)
        log_module_info(self.crit)
        if self.hps.clipped_double:
            log_module_info(self.twin)

    @beartype
    def norm_rets(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize if return normalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.standardize(x)
        return x

    @beartype
    def denorm_rets(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize if return denormalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.destandardize(x)
        return x

    @beartype
    def sample_trns_batch(self) -> dict[str, torch.Tensor]:
        """Sample (a) batch(es) of transitions from the replay buffer(s)"""
        assert self.replay_buffers is not None

        batches = defaultdict(list)
        for rb in self.replay_buffers:
            batch = rb.sample(self.hps.batch_size)
            for k, v in batch.items():
                batches[k].append(v)
        out = {}
        for k, v in batches.items():
            out[k], _ = pack(v, "* d")  # equiv to: rearrange(v, "n b d -> (n b) d")
        return out

    @beartype
    def predict(self, ob: np.ndarray, *, apply_noise: bool) -> np.ndarray:
        """Predict an action, with or without perturbation"""
        # create tensor from the state (`require_grad=False` by default)
        ob_tensor = torch.Tensor(ob).to(self.device)

        # predict an action
        ac_tensor = self.actr.act(ob_tensor)
        # if desired, add noise to the predicted action
        if apply_noise:
            # apply additive action noise once the action has been predicted,
            # in combination with parameter noise, or not.
            ac_tensor += self.ac_noise.generate()

        # place on cpu as a numpy array
        ac = ac_tensor.numpy(force=True)
        # clip the action to fit within the range from the environment
        ac.clip(-self.max_ac, self.max_ac)
        return ac

    @beartype
    def compute_losses(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       next_state: torch.Tensor,
                       next_action: torch.Tensor,
                       reward: torch.Tensor,
                       done: torch.Tensor,
                       td_len: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the critic and actor losses"""

        # compute qz estimates
        q = self.denorm_rets(self.crit(state, action))
        twin_q = self.denorm_rets(self.twin(state, action))

        # compute target qz estimate and same for twin
        q_prime = self.targ_crit(next_state, next_action)
        twin_q_prime = self.targ_twin(next_state, next_action)
        if self.hps.bcq_style_targ_mix:
            # use BCQ style of target mixing: soft minimum
            q_prime = (0.75 * torch.min(q_prime, twin_q_prime) +
                       0.25 * torch.max(q_prime, twin_q_prime))
        else:
            # use TD3 style of target mixing: hard minimum
            q_prime = torch.min(q_prime, twin_q_prime)

        q_prime = self.denorm_rets(q_prime)

        # assemble the Bellman target
        targ_q = (reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime)

        targ_q = targ_q.detach()

        targ_q = self.norm_rets(targ_q)
        if self.hps.ret_norm:
            # update the running stats
            self.rms_ret.update(targ_q)

        # critic and twin losses
        crit_loss = ff.smooth_l1_loss(q, targ_q)  # Huber loss for both here and below
        twin_loss = ff.smooth_l1_loss(twin_q, targ_q)  # overwrites the None initially set

        # actor loss
        actr_loss = -self.crit(state, self.actr.act(state))

        actr_loss = actr_loss.mean()

        return actr_loss, crit_loss, twin_loss

    @beartype
    @staticmethod
    def send_to_dash(metrics: dict[str, Union[np.float64, np.int64, np.ndarray]],
                     *,
                     step_metric: int,
                     glob: str):
        """Send the metrics to the wandb dashboard"""

        wandb_dict = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                assert v.ndim == 0
            assert hasattr(v, "item"), "in case of API changes"
            wandb_dict[f"{glob}/{k}"] = v.item()

        wandb_dict[f"{glob}/step"] = step_metric

        wandb.log(wandb_dict)
        logger.debug(f"logged this to wandb: {wandb_dict}")

    @beartype
    def update_actr_crit(self,
                         trns_batch: dict[str, torch.Tensor],
                         *,
                         update_actr: bool):
        """Update the critic and the actor"""

        with torch.no_grad():
            # define inputs
            state = trns_batch["obs0"]
            action = trns_batch["acs0"]
            next_state = trns_batch["obs1"]
            reward = trns_batch["erews1"]
            done = trns_batch["dones1"].float()
            td_len = torch.ones_like(done)
            # update the observation normalizer
            self.rms_obs.update(state)

        # compute target action
        if self.hps.targ_actor_smoothing:
            n_ = action.clone().detach().normal_(0., self.hps.td3_std).to(self.device)
            n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
            next_action = (
                self.targ_actr.act(next_state) + n_).clamp(-self.max_ac, self.max_ac)
        else:
            next_action = self.targ_actr.act(next_state)

        # compute critic and actor losses
        actr_loss, crit_loss, twin_loss = self.compute_losses(
            state, action, next_state, next_action, reward, done, td_len)

        if update_actr:
            # update actor
            self.actr_opt.zero_grad()
            actr_loss.backward()
            if self.hps.clip_norm > 0:
                cg.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
            self.actr_opt.step()

            self.actr_updates_so_far += 1

            if self.actr_updates_so_far % self.TRAIN_METRICS_WANDB_LOG_FREQ == 0:
                wandb_dict = {"actr_loss": actr_loss.numpy(force=True)}
                self.send_to_dash(
                    wandb_dict,
                    step_metric=self.actr_updates_so_far,
                    glob="train_actr",
                )

        # update critic
        self.crit_opt.zero_grad()
        crit_loss.backward()
        self.crit_opt.step()
        if twin_loss is not None:
            # update twin
            self.twin_opt.zero_grad()
            twin_loss.backward()
            self.twin_opt.step()

        self.crit_updates_so_far += 1

        if self.crit_updates_so_far % self.TRAIN_METRICS_WANDB_LOG_FREQ == 0:
            wandb_dict = {"crit_loss": crit_loss.numpy(force=True)}
            if twin_loss is not None:
                wandb_dict.update({"twin_loss": twin_loss.numpy(force=True)})
            self.send_to_dash(
                wandb_dict,
                step_metric=self.crit_updates_so_far,
                glob="train_crit",
            )

        # update target nets
        self.update_target_net()

    @beartype
    def update_target_net(self):
        """Update the target networks"""

        with torch.no_grad():
            for param, targ_param in zip(self.actr.parameters(),
                                         self.targ_actr.parameters()):
                new_param = self.hps.polyak * param
                new_param += (1. - self.hps.polyak) * targ_param
                targ_param.copy_(new_param)
            for param, targ_param in zip(self.crit.parameters(),
                                         self.targ_crit.parameters()):
                new_param = self.hps.polyak * param
                new_param += (1. - self.hps.polyak) * targ_param
                targ_param.copy_(new_param)
            if self.hps.clipped_double:
                for param, targ_param in zip(self.twin.parameters(),
                                             self.targ_twin.parameters()):
                    new_param = self.hps.polyak * param
                    new_param += (1. - self.hps.polyak) * targ_param
                    targ_param.copy_(new_param)

    @beartype
    def save(self, path: Path, sfx: Optional[str] = None):
        """Save the agent to disk and wandb servers"""
        # prep checkpoint
        fname = (f"ckpt_{sfx}"
                 if sfx is not None
                 else f".ckpt_{self.timesteps_so_far}ts")
        # design choice: hide the ckpt saved without an extra qualifier
        path = (parent := path) / f"{fname}.pth"
        checkpoint = {
            "hps": self.hps,  # handy for archeology
            "timesteps_so_far": self.timesteps_so_far,
            # and now the state_dict objects
            "rms_obs": self.rms_obs.state_dict(),
            "actr": self.actr.state_dict(),
            "crit": self.crit.state_dict(),
            "actr_opt": self.actr_opt.state_dict(),
            "crit_opt": self.crit_opt.state_dict(),
        }
        if self.hps.clipped_double:
            checkpoint.update({
                "twin": self.twin.state_dict(),
                "twin_opt": self.twin_opt.state_dict(),
            })
        # save checkpoint to filesystem
        torch.save(checkpoint, path)
        if sfx == "best":
            # upload the model to wandb servers
            wandb.save(str(path), base_path=parent)

    @beartype
    def load(self, wandb_run_path: str, model_name: str = "ckpt_best.pth"):
        """Download a model from wandb and load it"""
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file = run.file(model_name)
            file.download(root=tmp_dir_name, replace=True)
            tmp_file_path = Path(tmp_dir_name) / model_name
            # load the agent stored in this file
            self.load_from_disk(tmp_file_path)

    @beartype
    def load_from_disk(self, path: Path):
        """Load another agent into this one"""
        checkpoint = torch.load(path)
        if "timesteps_so_far" in checkpoint:
            self.timesteps_so_far = checkpoint["timesteps_so_far"]
        # the "strict" argument of `load_state_dict` is True by default
        self.rms_obs.load_state_dict(checkpoint["rms_obs"])
        self.actr.load_state_dict(checkpoint["actr"])
        self.crit.load_state_dict(checkpoint["crit"])
        self.actr_opt.load_state_dict(checkpoint["actr_opt"])
        self.crit_opt.load_state_dict(checkpoint["crit_opt"])
        if self.hps.clipped_double:
            if "twin" in checkpoint:
                self.twin.load_state_dict(checkpoint["twin"])
                if "twin_opt" in checkpoint:
                    self.twin_opt.load_state_dict(checkpoint["twin_opt"])
                else:
                    logger.warn("twin opt is missing from the loaded ckpt!")
                    logger.warn("we move on nonetheless, from a fresh opt")
            else:
                raise IOError("no twin found in checkpoint ckpt file")
        elif "twin" in checkpoint:  # in the case where clipped double is off
            logger.warn("there is a twin the loaded ckpt, but you want none")

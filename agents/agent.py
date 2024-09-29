import tempfile
from pathlib import Path
from typing import Optional, Union, Any
from collections import defaultdict

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
from einops import pack
import wandb
import numpy as np
import torch
import torch.special
from torch.optim import Adam
from torch.nn.utils import clip_grad as cg
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as ff

from helpers import logger
from helpers.normalizer import RunningMoments
from agents.nets import log_module_info, Actor, TanhGaussActor, Critic
from agents.memory import ReplayBuffer


class Agent(object):

    TRAIN_METRICS_WANDB_LOG_FREQ: int = 100

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 min_ac: np.ndarray,
                 max_ac: np.ndarray,
                 device: torch.device,
                 hps: DictConfig,
                 actr_noise_rng: torch.Generator,
                 replay_buffers: Optional[list[ReplayBuffer]]):
        self.ob_shape, self.ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]

        self.device = device

        self.min_ac = torch.tensor(min_ac, dtype=torch.float32, device=self.device)
        self.max_ac = torch.tensor(max_ac, dtype=torch.float32, device=self.device)

        assert isinstance(hps, DictConfig)
        self.hps = hps

        self.ctx = (
            autocast(
                enabled=self.hps.cuda,
                dtype=torch.float16 if self.hps.fp16 else torch.float32,
            )
        )
        self.timesteps_so_far = 0
        self.actr_updates_so_far = 0
        self.crit_updates_so_far = 0

        self.best_eval_ep_ret = -float("inf")  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay buffer
        self.replay_buffers = replay_buffers

        self.rms_obs = None
        if self.hps.batch_norm:
            # create observation normalizer that maintains running statistics
            self.rms_obs = RunningMoments(shape=self.ob_shape, device=self.device)

        # create online and target nets

        actr_net_args = [self.ob_shape,
                         self.ac_shape,
                         (256, 256),
                         self.rms_obs,
                         self.min_ac,
                         self.max_ac]
        actr_net_kwargs = {"layer_norm": self.hps.layer_norm,
                           "device": self.device}
        if not self.hps.prefer_td3_over_sac:
            actr_net_kwargs.update({"generator": actr_noise_rng})
            actr_module = TanhGaussActor
        else:
            actr_net_kwargs.update({"exploration_noise": self.hps.actr_noise_std})
            actr_module = Actor
        self.actr = actr_module(*actr_net_args, **actr_net_kwargs)
        if self.hps.prefer_td3_over_sac:
            # using TD3 (SAC does not use a target actor)
            self.targ_actr = Actor(*actr_net_args, **actr_net_kwargs)
            # note: could use `actr_module` equivalently, but prefer most explicit

        crit_net_args = [self.ob_shape,
                         self.ac_shape,
                         (256, 256),
                         self.rms_obs]
        crit_net_kwargs = {"layer_norm": self.hps.layer_norm,
                           "device": self.device}
        self.crit = Critic(*crit_net_args, **crit_net_kwargs)
        self.twin = Critic(*crit_net_args, **crit_net_kwargs)
        self.targ_crit = Critic(*crit_net_args, **crit_net_kwargs)
        self.targ_twin = Critic(*crit_net_args, **crit_net_kwargs)

        # initilize the target nets
        if self.hps.prefer_td3_over_sac:
            # using TD3 (SAC does not use a target actor)
            self.targ_actr.load_state_dict(self.actr.state_dict())
        self.targ_crit.load_state_dict(self.crit.state_dict())
        self.targ_twin.load_state_dict(self.twin.state_dict())

        # set up the optimizers
        self.actr_opt = Adam(self.actr.parameters(), lr=self.hps.actr_lr)
        self.crit_opt = Adam(self.crit.parameters(), lr=self.hps.crit_lr)
        self.twin_opt = Adam(self.twin.parameters(), lr=self.hps.crit_lr)

        # set up the gradient scalers
        self.actr_sclr = GradScaler(enabled=self.hps.fp16)
        self.crit_sclr = GradScaler(enabled=self.hps.fp16)
        self.twin_sclr = GradScaler(enabled=self.hps.fp16)
        if not self.hps.prefer_td3_over_sac:
            self.loga_sclr = GradScaler(enabled=self.hps.fp16)

        if not self.hps.prefer_td3_over_sac:
            # setup log(alpha) if SAC is chosen
            self.log_alpha = torch.as_tensor(self.hps.alpha_init, device=self.device).log()

            if self.hps.autotune:
                # create learnable Lagrangian multiplier
                # common trick: learn log(alpha) instead of alpha directly
                self.log_alpha.requires_grad = True
                self.targ_ent = -self.ac_shape[-1]  # set target entropy to -|A|
                self.loga_opt = Adam([self.log_alpha], lr=self.hps.log_alpha_lr)

        # log module architectures
        log_module_info(self.actr)
        log_module_info(self.crit)
        log_module_info(self.twin)

    @property
    def alpha(self) -> Optional[torch.Tensor]:
        if not self.hps.prefer_td3_over_sac:
            return self.log_alpha.exp()
        return None

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
    def predict(self, ob: torch.Tensor, *, explore: bool) -> np.ndarray:
        """Predict an action, with or without perturbation"""
        if self.hps.prefer_td3_over_sac:
            with torch.no_grad():
                # using TD3
                ac = self.actr(ob) if explore else self.actr.explore(ob)
        else:
            # using SAC
            # actions from sample and mode are detached by default
            ac = (self.actr.sample(ob) if explore else self.actr.mode(ob))
        return ac.clamp(self.min_ac, self.max_ac).cpu().numpy()

    @beartype
    def build_loss_operands(self, trns_batch: dict[str, torch.Tensor]):

        with torch.no_grad():
            # define inputs
            state = trns_batch["obs0"]
            action = trns_batch["acs0"]
            next_state = trns_batch["obs1"]
            reward = trns_batch["erews1"]
            done = trns_batch["dones1"].float()
            td_len = torch.ones_like(done)

            if self.hps.batch_norm:
                assert self.rms_obs is not None
                # update the observation normalizer
                self.rms_obs.update(state)

        # compute target action
        if self.hps.prefer_td3_over_sac:
            # using TD3
            if self.hps.targ_actor_smoothing:
                n_ = action.clone().detach().normal_(0., self.hps.td3_std)
                assert n_.device == self.device
                n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
                next_action = (
                    self.targ_actr(next_state) + n_).clamp(self.min_ac, self.max_ac)
            else:
                next_action = self.targ_actr(next_state)
        else:
            # using SAC
            next_action = self.actr.sample(next_state, stop_grad=True)

        return state, action, next_state, next_action, reward, done, td_len

    @beartype
    def compute_losses(self,
                       state: torch.Tensor,
                       action: torch.Tensor,
                       next_state: torch.Tensor,
                       next_action: torch.Tensor,
                       reward: torch.Tensor,
                       done: torch.Tensor,
                       td_len: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the critic and actor losses"""

        loga_loss = None  # not used if using TD3

        if self.hps.prefer_td3_over_sac:
            # using TD3
            action_from_actr = self.actr(state)
            log_prob = None  # quiets down the type checker
        else:
            # using SAC
            action_from_actr = self.actr.sample(state, stop_grad=False)
            log_prob = self.actr.logp(state, action_from_actr)
            # here, there are two gradient pathways: the reparam trick makes the sampling process
            # differentiable (pathwise derivative), and logp is a score function gradient estimator
            # intuition: aren't they competing and therefore messing up with each other's compute
            # graphs? to understand what happens, write down the closed form of the Normal's logp
            # (or see this formula in nets.py) and replace x by mean + eps * std
            # it shows that with both of these gradient pathways, the mean receives no gradient
            # only the std receives some (they cancel out)
            # moreover, if only the std receives gradient, we can expect subpar results if this std
            # is state independent
            # this can be observed here, and has been noted in openai/spinningup
            # in native PyTorch, it is equivalent to using `log_prob` on a sample from `rsample`
            # note also that detaching the action in the logp (using `sample`, and not `rsample`)
            # yields to poor results, showing how allowing for non-zero gradients for the mean
            # can have a destructive effect, and that is why SAC does not allow them to flow.

        # compute qz estimates
        q = self.crit(state, action)
        twin_q = self.twin(state, action)

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

        if not self.hps.prefer_td3_over_sac:  # only for SAC
            assert self.alpha is not None
            # add the causal entropy regularization term
            next_log_prob = self.actr.logp(next_state, next_action)
            q_prime -= self.alpha.detach() * next_log_prob

        # assemble the Bellman target
        targ_q = (reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime)

        targ_q = targ_q.detach()

        # critic and twin losses
        crit_loss = ff.mse_loss(q, targ_q)
        twin_loss = ff.mse_loss(twin_q, targ_q)

        # actor loss
        if self.hps.prefer_td3_over_sac:
            actr_loss = -self.crit(state, action_from_actr)
        else:
            assert self.alpha is not None
            actr_loss = (self.alpha.detach() * log_prob) - torch.min(
                self.crit(state, action_from_actr),
                self.twin(state, action_from_actr))
            if not actr_loss.mean().isfinite():
                raise ValueError("NaNs: numerically unstable arctanh func")
        actr_loss = actr_loss.mean()

        if (not self.hps.prefer_td3_over_sac) and self.hps.autotune:
            assert log_prob is not None
            loga_loss = (self.log_alpha * (-log_prob - self.targ_ent).detach()).mean()

        return actr_loss, crit_loss, twin_loss, loga_loss

    @beartype
    def update_actr(self, actr_loss: torch.Tensor, loga_loss: Optional[torch.Tensor]):

        # update actor
        self.actr_opt.zero_grad()
        actr_loss = self.actr_sclr.scale(actr_loss)
        actr_loss.backward()
        if self.hps.clip_norm > 0:
            self.actr_sclr.unscale_(self.actr_opt)
            cg.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.actr_sclr.step(self.actr_opt)
        self.actr_sclr.update()

        if loga_loss is not None:
            # update log(alpha), and therefore alpha
            assert (not self.hps.prefer_td3_over_sac) and self.hps.autotune
            self.loga_opt.zero_grad()
            loga_loss = self.loga_sclr.scale(loga_loss)
            assert loga_loss is not None
            loga_loss.backward()
            self.loga_sclr.step(self.loga_opt)
            self.loga_sclr.update()

    @beartype
    def update_crit(self, crit_loss: torch.Tensor, twin_loss: torch.Tensor):

        # update critic
        self.crit_opt.zero_grad()
        crit_loss = self.crit_sclr.scale(crit_loss)
        crit_loss.backward()
        self.crit_sclr.step(self.crit_opt)
        self.crit_sclr.update()
        # update twin
        self.twin_opt.zero_grad()
        twin_loss = self.twin_sclr.scale(twin_loss)
        twin_loss.backward()
        self.twin_sclr.step(self.twin_opt)
        self.twin_sclr.update()

    @beartype
    def update_targ_nets(self):

        if (self.hps.prefer_td3_over_sac or (
            self.crit_updates_so_far % self.hps.crit_targ_update_freq == 0)):

            with torch.no_grad():
                if self.hps.prefer_td3_over_sac:
                    # using TD3 (SAC does not use a target actor)
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
            "actr": self.actr.state_dict(),
            "crit": self.crit.state_dict(),
            "twin": self.twin.state_dict(),
            "actr_opt": self.actr_opt.state_dict(),
            "crit_opt": self.crit_opt.state_dict(),
            "twin_opt": self.twin_opt.state_dict(),
        }
        if self.hps.batch_norm:
            assert self.rms_obs is not None
            checkpoint.update({
                "rms_obs": self.rms_obs.state_dict()})
        # save checkpoint to filesystem
        torch.save(checkpoint, path)
        logger.info(f"{sfx} model saved to disk")
        if sfx == "best":
            # upload the model to wandb servers
            wandb.save(str(path), base_path=parent)
            logger.info("model saved to wandb")

    @beartype
    def load_from_disk(self, path: Path):
        """Load another agent into this one"""
        checkpoint = torch.load(path)
        if "timesteps_so_far" in checkpoint:
            self.timesteps_so_far = checkpoint["timesteps_so_far"]
        # the "strict" argument of `load_state_dict` is True by default
        if self.hps.batch_norm:
            assert self.rms_obs is not None
            self.rms_obs.load_state_dict(checkpoint["rms_obs"])
        self.actr.load_state_dict(checkpoint["actr"])
        self.crit.load_state_dict(checkpoint["crit"])
        self.actr_opt.load_state_dict(checkpoint["actr_opt"])
        self.crit_opt.load_state_dict(checkpoint["crit_opt"])
        if "twin" in checkpoint:
            self.twin.load_state_dict(checkpoint["twin"])
            if "twin_opt" in checkpoint:
                self.twin_opt.load_state_dict(checkpoint["twin_opt"])
            else:
                logger.info("twin opt is missing from the loaded ckpt!")
                logger.info("we move on nonetheless, from a fresh opt")
        else:
            raise IOError("no twin found in checkpoint ckpt file")

    @beartype
    @staticmethod
    def compare_dictconfigs(
        dictconfig1: DictConfig,
        dictconfig2: DictConfig,
    ) -> dict[str, dict[str, Union[str, int, list[int], dict[str, Union[str, int, list[int]]]]]]:
        """Compare two DictConfig objects and return the differences.
        Returns a dictionary with keys "added", "removed", and "changed".
        """
        differences = {"added": {}, "removed": {}, "changed": {}}

        keys1 = set(dictconfig1.keys())
        keys2 = set(dictconfig2.keys())

        # added keys
        for key in keys2 - keys1:
            differences["added"][key] = dictconfig2[key]

        # removed keys
        for key in keys1 - keys2:
            differences["removed"][key] = dictconfig1[key]

        # changed keys
        for key in keys1 & keys2:
            if dictconfig1[key] != dictconfig2[key]:
                differences["changed"][key] = {
                    "from": dictconfig1[key], "to": dictconfig2[key]}

        return differences

    @beartype
    def load(self, wandb_run_path: str, model_name: str = "ckpt_best.pth"):
        """Download a model from wandb and load it"""
        api = wandb.Api()
        run = api.run(wandb_run_path)
        # compare the current cfg with the cfg of the loaded model
        wandb_cfg_dict: dict[str, Any] = run.config
        wandb_cfg: DictConfig = OmegaConf.create(wandb_cfg_dict)
        a, r, c = self.compare_dictconfigs(wandb_cfg, self.hps).values()
        # N.B.: in Python 3.7 and later, dicts preserve the insertion order
        logger.warn(f"added  : {a}")
        logger.warn(f"removed: {r}")
        logger.warn(f"changed: {c}")
        # create a temporary directory to download to
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            file = run.file(model_name)
            # download the model file from wandb servers
            file.download(root=tmp_dir_name, replace=True)
            logger.warn("model downloaded from wandb to disk")
            tmp_file_path = Path(tmp_dir_name) / model_name
            # load the agent stored in this file
            self.load_from_disk(tmp_file_path)
            logger.warn("model loaded")

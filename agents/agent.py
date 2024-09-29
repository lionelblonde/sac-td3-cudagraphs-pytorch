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
from torch.nn import functional as ff
from tensordict import TensorDict

from helpers import logger
from helpers.normalizer import RunningMoments
from agents.nets import log_module_info, Actor, TanhGaussActor, Critic
from agents.memory import ReplayBuffer


class Agent(object):

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 min_ac: np.ndarray,
                 max_ac: np.ndarray,
                 device: torch.device,
                 hps: DictConfig,
                 generator: torch.Generator,
                 replay_buffers: Optional[list[ReplayBuffer]]):
        self.ob_shape, self.ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]

        self.device = device

        self.min_ac = torch.tensor(min_ac, dtype=torch.float32, device=self.device)
        self.max_ac = torch.tensor(max_ac, dtype=torch.float32, device=self.device)

        assert isinstance(hps, DictConfig)
        self.hps = hps

        self.timesteps_so_far = 0
        self.actor_updates_so_far = 0
        self.qnet_updates_so_far = 0

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

        actor_net_args = [self.ob_shape,
                          self.ac_shape,
                          (256, 256),
                          self.rms_obs,
                          self.min_ac,
                          self.max_ac]
        actor_net_kwargs = {"layer_norm": self.hps.layer_norm}
        if self.hps.prefer_td3_over_sac:
            actor_net_kwargs.update({"exploration_noise": self.hps.actor_noise_std})
        else:
            actor_net_kwargs.update({"generator": generator})

        self.actor = (Actor if self.hps.prefer_td3_over_sac else TanhGaussActor)(
            *actor_net_args, **actor_net_kwargs, device=self.device)
        self.actor_params = TensorDict.from_module(self.actor, as_module=True)
        assert self.actor_params.data is not None
        self.actor_target = self.actor_params.data.clone()
        # discard params of net
        self.actor = (Actor if self.hps.prefer_td3_over_sac else TanhGaussActor)(
            *actor_net_args, **actor_net_kwargs, device="meta")
        self.actor_params.to_module(self.actor)

        qnet_net_args = [self.ob_shape,
                         self.ac_shape,
                         (256, 256),
                         self.rms_obs]
        qnet_net_kwargs = {"layer_norm": self.hps.layer_norm}

        self.qnet1 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        self.qnet2 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        self.qnet_params = TensorDict.from_modules(self.qnet1, self.qnet2, as_module=True)
        assert self.qnet_params.data is not None
        self.qnet_target = self.qnet_params.data.clone()
        # discard params of net
        self.qnet = Critic(*qnet_net_args, **qnet_net_kwargs, device="meta")
        self.qnet_params.to_module(self.qnet)

        # set up the optimizers

        self.q_optimizer = Adam(
            self.qnet.parameters(),
            lr=self.hps.crit_lr)  # capturable=args.cudagraphs and not args.compile)
        self.actor_optimizer = Adam(
            list(self.actor.parameters()),
            lr=self.hps.actor_lr)  # capturable=args.cudagraphs and not args.compile)

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
        log_module_info(self.actor)
        log_module_info(self.qnet1)
        log_module_info(self.qnet2)

    # TODO(lionel): beartype this
    def batched_qf(self, params, ob, action, next_q_value=None):
        with params.to_module(self.qnet):
            vals = self.qnet(ob, action)
            if next_q_value is not None:
                return ff.mse_loss(vals.view(-1), next_q_value)
            return vals

    # TODO(lionel): beartype this
    def pi(self, params, ob):
        with params.to_module(self.actor):
            return self.actor(ob)

    @beartype
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
                ac = self.actor(ob) if explore else self.actor.explore(ob)
        else:
            # using SAC
            # actions from sample and mode are detached by default
            ac = (self.actor.sample(ob) if explore else self.actor.mode(ob))
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
            pi_next_target = self.pi(self.actor_target, next_state)  # target actor
            if self.hps.targ_actor_smoothing:
                n_ = action.clone().detach().normal_(0., self.hps.td3_std)
                assert n_.device == self.device
                n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
                next_action = (pi_next_target + n_).clamp(self.min_ac, self.max_ac)
            else:
                next_action = pi_next_target
        else:
            # using SAC
            next_action = self.actor.sample(next_state, stop_grad=True)

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
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute the critic and actor losses"""

        loga_loss = None  # not used if using TD3

        if self.hps.prefer_td3_over_sac:
            # using TD3
            action_from_actor = self.actor(state)
            log_prob = None  # quiets down the type checker
        else:
            # using SAC
            action_from_actor = self.actor.sample(state, stop_grad=False)
            log_prob = self.actor.logp(state, action_from_actor)
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

        self.q_optimizer.zero_grad()

        with torch.no_grad():
            qf_next_target = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_target, next_state, next_action,
            )

            qf_min = qf_next_target.min(0).values
            if self.hps.bcq_style_targ_mix:
                # use BCQ style of target mixing: soft minimum
                qf_max = qf_next_target.max(0).values
                q_prime = ((0.75 * qf_min) + (0.25 * qf_max))
            else:
                # use TD3 style of target mixing: hard minimum
                q_prime = qf_min

            if not self.hps.prefer_td3_over_sac:  # only for SAC
                assert self.alpha is not None
                # add the causal entropy regularization term
                next_log_prob = self.actor.logp(next_state, next_action)
                q_prime -= self.alpha.detach() * next_log_prob

            # assemble the Bellman target
            targ_q = (reward + (self.hps.gamma ** td_len) * (1. - done) * q_prime)
            targ_q = targ_q.squeeze()

        qf_a_values = torch.vmap(self.batched_qf, (0, None, None, None))(
            self.qnet_params, state, action, targ_q,
        )
        qf_loss = qf_a_values.sum(0)

        # actor loss
        self.actor_optimizer.zero_grad()

        if self.hps.prefer_td3_over_sac:
            qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_params.data, state, action_from_actor)
            min_qf_pi = qf_pi[0]
            actor_loss = -min_qf_pi
        else:
            assert self.alpha is not None
            qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_params.data, state, action_from_actor)
            min_qf_pi = qf_pi.min(0).values
            actor_loss = (self.alpha.detach() * log_prob) - min_qf_pi
            if not actor_loss.mean().isfinite():
                raise ValueError("NaNs: numerically unstable arctanh func")
        actor_loss = actor_loss.mean()

        if (not self.hps.prefer_td3_over_sac) and self.hps.autotune:
            assert log_prob is not None
            self.loga_opt.zero_grad()
            loga_loss = (self.log_alpha * (-log_prob - self.targ_ent).detach()).mean()

        return actor_loss, qf_loss, loga_loss

    @beartype
    def update_actor(self, actor_loss: torch.Tensor, loga_loss: Optional[torch.Tensor]):

        actor_loss.backward()
        if self.hps.clip_norm > 0:
            cg.clip_grad_norm_(self.actor.parameters(), self.hps.clip_norm)
        self.actor_optimizer.step()

        if loga_loss is not None:
            # update alpha
            assert (not self.hps.prefer_td3_over_sac) and self.hps.autotune
            loga_loss.backward()
            self.loga_opt.step()

    @beartype
    def update_crit(self, qf_loss: torch.Tensor):
        qf_loss.backward()
        self.q_optimizer.step()

    @beartype
    def update_targ_nets(self):

        if (self.hps.prefer_td3_over_sac or (
            self.qnet_updates_so_far % self.hps.crit_targ_update_freq == 0)):

            # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y

            self.qnet_target.lerp_(self.qnet_params.data, self.hps.polyak)
            if self.hps.prefer_td3_over_sac:
                # using TD3 (SAC does not use a target actor)
                self.actor_target.lerp_(self.actor_params.data, self.hps.polyak)

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
            "actor": self.actor.state_dict(),
            "qnet1": self.qnet1.state_dict(),
            "qnet2": self.qnet2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
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
        self.actor.load_state_dict(checkpoint["actor"])
        self.qnet1.load_state_dict(checkpoint["qnet1"])
        self.qnet2.load_state_dict(checkpoint["qnet2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

    @beartype
    @staticmethod
    def compare_dictconfigs(
        dictconfig1: DictConfig,
        dictconfig2: DictConfig,
    ) -> dict[str, dict[str, Union[str, int, list[int], dict[str, Union[str, int, list[int]]]]]]:
        """Compare two DictConfig objects of depth=1 and return the differences.
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

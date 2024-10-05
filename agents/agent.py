import tempfile
from pathlib import Path
from typing import Optional, Union, Any

from beartype import beartype
from omegaconf import OmegaConf, DictConfig
import wandb
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.nn import functional as ff
from torch.nn.utils import clip_grad
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import ReplayBuffer

from helpers import logger
from agents.nets import log_module_info, Actor, TanhGaussActor, Critic


class Agent(object):

    @beartype
    def __init__(self,
                 net_shapes: dict[str, tuple[int, ...]],
                 min_ac: np.ndarray,
                 max_ac: np.ndarray,
                 device: torch.device,
                 hps: DictConfig,
                 generator: torch.Generator,
                 rb: Optional[ReplayBuffer] = None):
        self.ob_shape, self.ac_shape = net_shapes["ob_shape"], net_shapes["ac_shape"]

        self.device = device

        self.min_ac = torch.tensor(min_ac, dtype=torch.float32, device=self.device)
        self.max_ac = torch.tensor(max_ac, dtype=torch.float32, device=self.device)

        assert isinstance(hps, DictConfig)
        self.hps = hps

        self.ctx = (
            torch.autocast(
                "cuda",
                enabled=self.hps.cuda,
                dtype=torch.bfloat16 if self.hps.bfloat16 else torch.float32,
            )
        )

        self.timesteps_so_far = 0
        self.actor_updates_so_far = 0
        self.qnet_updates_so_far = 0

        self.best_eval_ep_ret = -float("inf")  # updated in orchestrator

        assert self.hps.segment_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm <= 0, hence disabled")

        # replay buffer
        self.rb = rb

        # create online and target nets

        actor_net_args = [
            self.ob_shape, self.ac_shape, (256, 256), self.min_ac, self.max_ac]
        actor_net_kwargs = {"layer_norm": self.hps.layer_norm}
        if self.hps.prefer_td3_over_sac:
            actor_net_kwargs.update({"exploration_noise": self.hps.actor_noise_std})
        else:
            actor_net_kwargs.update({"generator": generator})

        self.actor = (Actor if self.hps.prefer_td3_over_sac else TanhGaussActor)(
            *actor_net_args, **actor_net_kwargs, device=self.device)
        self.actor_params = TensorDict.from_module(self.actor, as_module=True)
        self.actor_target = self.actor_params.data.clone()

        # discard params of net
        self.actor = (Actor if self.hps.prefer_td3_over_sac else TanhGaussActor)(
            *actor_net_args, **actor_net_kwargs, device="meta")
        self.actor_params.to_module(self.actor)

        self.actor_detach = (Actor if self.hps.prefer_td3_over_sac else TanhGaussActor)(
            *actor_net_args, **actor_net_kwargs, device=self.device)
        # copy params to actor_detach without grad
        TensorDict.from_module(self.actor).data.to_module(self.actor_detach)
        if self.hps.prefer_td3_over_sac:
            self.policy = TensorDictModule(
                self.actor_detach,
                in_keys=["observation"],
                out_keys=["action"],
            )
            self.policy_explore = TensorDictModule(
                self.actor_detach.explore,
                in_keys=["observation"],
                out_keys=["action"],
            )
        else:
            self.policy = TensorDictModule(
                self.actor_detach.get_action,
                in_keys=["observation"],
                out_keys=["mean"],  # mode
            )
            self.policy_explore = TensorDictModule(
                self.actor_detach.get_action,
                in_keys=["observation"],
                out_keys=["action"],  # sample
            )

        if self.hps.compile:
            self.policy = torch.compile(self.policy, mode=None)
            self.policy_explore = torch.compile(self.policy_explore, mode=None)

        qnet_net_args = [self.ob_shape, self.ac_shape, (256, 256)]
        qnet_net_kwargs = {"layer_norm": self.hps.layer_norm}

        self.qnet1 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        self.qnet2 = Critic(*qnet_net_args, **qnet_net_kwargs, device=self.device)
        self.qnet_params = TensorDict.from_modules(self.qnet1, self.qnet2, as_module=True)
        self.qnet_target = self.qnet_params.data.clone()

        # discard params of net
        self.qnet = Critic(*qnet_net_args, **qnet_net_kwargs, device="meta")
        self.qnet_params.to_module(self.qnet)

        # set up the optimizers

        self.q_optimizer = Adam(
            self.qnet.parameters(),
            lr=self.hps.crit_lr,
            capturable=self.hps.cudagraphs and not self.hps.compile,
        )
        self.actor_optimizer = Adam(
            list(self.actor.parameters()),
            lr=self.hps.actor_lr,
            capturable=self.hps.cudagraphs and not self.hps.compile,
        )

        if not self.hps.prefer_td3_over_sac:
            # setup log(alpha) if SAC is chosen
            self.log_alpha = torch.as_tensor(self.hps.alpha_init, device=self.device).log()

            if self.hps.autotune:
                # create learnable Lagrangian multiplier
                # common trick: learn log(alpha) instead of alpha directly
                self.log_alpha.requires_grad = True
                self.targ_ent = -self.ac_shape[-1]  # set target entropy to -|A|
                self.alpha_opt = Adam(
                    [self.log_alpha],
                    lr=self.hps.log_alpha_lr,
                    capturable=self.hps.cudagraphs and not self.hps.compile,
                )

        # log module architectures
        log_module_info(self.actor)
        log_module_info(self.qnet1)
        log_module_info(self.qnet2)

    @beartype
    def batched_qf(self,
                   params: TensorDict,
                   ob: torch.Tensor,
                   action: torch.Tensor,
                   next_q_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Use two qnet networks from params"""
        with params.to_module(self.qnet):
            vals = self.qnet(ob, action)
            if next_q_value is not None:
                return ff.mse_loss(vals.view(-1), next_q_value)
            return vals

    @beartype
    def pi(self, params: TensorDict, ob: torch.Tensor) -> torch.Tensor:
        """Use an actor network from params"""
        with params.to_module(self.actor):
            return self.actor(ob)

    @property
    @beartype
    def alpha(self) -> Optional[torch.Tensor]:
        if not self.hps.prefer_td3_over_sac:
            return self.log_alpha.exp()
        return None

    @beartype
    def predict(self, state: torch.Tensor, *, explore: bool) -> np.ndarray:
        """Predict an action, with or without perturbation"""
        action = self.policy_explore(state) if explore else self.policy(state)
        if self.hps.prefer_td3_over_sac:
            action.clamp(self.min_ac, self.max_ac)
        return action.cpu().numpy()

    @beartype
    def update_qnets(self, batch: TensorDict) -> TensorDict:

        self.q_optimizer.zero_grad()

        with torch.no_grad():

            # compute target action
            if self.hps.prefer_td3_over_sac:
                # using TD3
                next_state_log_pi = None
                pi_next_target = self.pi(
                    self.actor_target, batch["next_observations"])  # target actor
                # why use `pi`: we only have a handle on the target actor parameters
                if self.hps.targ_actor_smoothing:
                    n_ = batch["actions"].clone().detach().normal_(0., self.hps.td3_std)
                    n_ = n_.clamp(-self.hps.td3_c, self.hps.td3_c)
                    next_action = (pi_next_target + n_).clamp(self.min_ac, self.max_ac)
                else:
                    next_action = pi_next_target
            else:
                # using SAC
                next_action, next_state_log_pi, _ = self.actor.get_action(
                    batch["next_observations"])

            qf_next_target = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_target, batch["next_observations"], next_action,
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
                # add the causal entropy regularization term
                q_prime -= self.alpha * next_state_log_pi

            # assemble the Bellman target
            targ_q = batch["rewards"].flatten() + (
                ~batch["dones"].flatten()
            ).float() * self.hps.gamma * q_prime.view(-1)

        qf_a_values = torch.vmap(self.batched_qf, (0, None, None, None))(
            self.qnet_params, batch["observations"], batch["actions"], targ_q,
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        self.q_optimizer.step()

        self.qnet_updates_so_far += 1

        return TensorDict(
            {
                "loss/qf_loss": qf_loss.detach(),
            },
        )

    @beartype
    def update_actor(self, batch: TensorDict) -> TensorDict:

        self.actor_optimizer.zero_grad()

        if self.hps.prefer_td3_over_sac:
            # using TD3
            action_from_actor = self.actor(batch["observations"])
        else:
            # using SAC
            action_from_actor, state_log_pi, _ = self.actor.get_action(batch["observations"])
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

        if self.hps.prefer_td3_over_sac:
            qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_params.data, batch["observations"], action_from_actor)
            min_qf_pi = qf_pi[0]
            actor_loss = -min_qf_pi
        else:
            qf_pi = torch.vmap(self.batched_qf, (0, None, None))(
                self.qnet_params.data, batch["observations"], action_from_actor)
            min_qf_pi = qf_pi.min(0).values
            actor_loss = (self.alpha.detach() * state_log_pi) - min_qf_pi
        actor_loss = actor_loss.mean()

        actor_loss.backward()
        if self.hps.clip_norm > 0:
            clip_grad.clip_grad_norm_(self.actor.parameters(), self.hps.clip_norm)
        self.actor_optimizer.step()

        self.actor_updates_so_far += 1

        if self.hps.prefer_td3_over_sac:
            return TensorDict(
                {
                    "loss/actor": actor_loss.detach(),
                },
            )

        if self.hps.autotune:
            self.alpha_opt.zero_grad()
            with torch.no_grad():
                _, state_log_pi, _ = self.actor.get_action(batch["observations"])
            alpha_loss = (self.alpha * (-state_log_pi - self.targ_ent).detach()).mean()  # alpha

            alpha_loss.backward()
            self.alpha_opt.step()

            return TensorDict(
                {
                    "loss/actor_loss": actor_loss.detach(),
                    "loss/alpha_loss": alpha_loss.detach(),
                    "vitals/alpha": self.alpha.detach(),
                },
            )

        return TensorDict(
            {
                "loss/actor_loss": actor_loss.detach(),
                "vitals/alpha": self.alpha.detach(),
            },
        )

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
        self.actor.load_state_dict(checkpoint["actor"])
        self.qnet1.load_state_dict(checkpoint["qnet1"])
        self.qnet2.load_state_dict(checkpoint["qnet2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

    @staticmethod
    @beartype
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

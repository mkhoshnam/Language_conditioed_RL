import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


SKILL_SLICE = slice(-3, -1)  # env obs tail: [place_onehot, stack_onehot, dest_h]


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = x.reshape(-1, self.mean.shape[0])
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_n
        self.mean = self.mean + delta * b_n / total
        self.var = (
            self.var * self.count
            + b_var * b_n
            + delta**2 * self.count * b_n / total
        ) / total
        self.count = total

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        out = np.clip((arr - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)

        # Critical for dual-head PPO:
        # the policy head selector uses the skill one-hot at obs[-3:-1].
        # If we normalize it, it stops being a clean one-hot.
        if out.shape[-1] >= 3:
            out[..., SKILL_SLICE] = arr[..., SKILL_SLICE]

        return out.astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n_skills=2):
        super().__init__()
        self.n_skills = n_skills
        self.act_dim = act_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Separate endpoint policies.
        self.actor_mean = nn.ModuleList(
            [nn.Linear(hidden, act_dim) for _ in range(n_skills)]
        )

        # Separate value functions. This avoids mixing place and stack returns.
        self.critic = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(n_skills)]
        )

        # Separate exploration scale per skill.
        self.actor_logstd = nn.Parameter(torch.full((n_skills, act_dim), -0.5))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        for head in self.actor_mean:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def _skill_weights(self, x):
        # x is normalized obs, but RunningMeanStd preserves raw skill one-hot.
        w = x[..., SKILL_SLICE]
        w = torch.clamp(w, 0.0, 1.0)

        denom = w.sum(dim=-1, keepdim=True)
        fallback = torch.zeros_like(w)
        fallback[..., 0] = 1.0  # safe fallback = place head

        return torch.where(
            denom > 0.1,
            w / denom.clamp_min(1e-6),
            fallback,
        )

    def forward(self, x):
        feat = self.shared(x)
        w = self._skill_weights(x)

        means = torch.stack(
            [head(feat) for head in self.actor_mean],
            dim=-1,
        )
        values = torch.stack(
            [head(feat).squeeze(-1) for head in self.critic],
            dim=-1,
        )

        mean = (means * w.unsqueeze(-2)).sum(dim=-1)
        value = (values * w).sum(dim=-1)

        base_logstd = self.actor_logstd.clamp(-5.0, 0.0)
        logstd = w @ base_logstd

        # Keep gripper exploration active for release actions.
        logstd = torch.cat(
            [logstd[..., :-1], logstd[..., -1:].clamp(min=-1.0)],
            dim=-1,
        )

        std = logstd.exp().expand_as(mean)
        return mean, std, value

    @staticmethod
    def _squashed_log_prob(dist, raw_action, action):
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(-1)

    def act(self, obs):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        logp = self._squashed_log_prob(dist, raw_action, action)
        return action, logp, value

    def deterministic_act(self, obs):
        mean, _, value = self(obs)
        return torch.tanh(mean), value

    def evaluate(self, obs, action):
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(action)
        logp = self._squashed_log_prob(dist, raw_action, action)
        entropy = dist.entropy().sum(-1)
        return logp, value, entropy


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim, device):
        self.n = n_steps
        self.device = device
        self.obs = torch.zeros(n_steps, obs_dim, device=device)
        self.raw_obs = torch.zeros(n_steps, obs_dim, device=device)
        self.acts = torch.zeros(n_steps, act_dim, device=device)
        self.logps = torch.zeros(n_steps, device=device)
        self.rews = torch.zeros(n_steps, device=device)
        self.vals = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.ptr = 0

    def add(self, obs, raw_obs, act, logp, rew, val, done):
        i = self.ptr
        self.obs[i] = obs
        self.raw_obs[i] = raw_obs
        self.acts[i] = act
        self.logps[i] = logp
        self.rews[i] = rew
        self.vals[i] = val
        self.dones[i] = done
        self.ptr += 1

    def full(self):
        return self.ptr >= self.n

    def reset(self):
        self.ptr = 0

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        advantages = torch.zeros_like(self.rews)
        last_gae = 0.0

        for t in reversed(range(self.n)):
            next_val = last_value if t == self.n - 1 else self.vals[t + 1]
            next_done = self.dones[t]
            delta = self.rews[t] + gamma * next_val * (1 - next_done) - self.vals[t]
            last_gae = delta + gamma * lam * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.vals
        return advantages, returns


class PPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=3e-4,
        n_steps=4096,
        n_epochs=8,
        batch_size=256,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.003,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        device="cpu",
    ):
        self.device = device
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.extra = {}

        self.anchor = None
        self.anchor_beta = float(os.environ.get("ANCHOR_BETA", "0.0"))
        self.freeze_obs_rms = False

        self.policy = ActorCritic(obs_dim, act_dim).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim, device)
        self.obs_rms = RunningMeanStd((obs_dim,))

    def _make_anchor_if_enabled(self):
        if self.anchor_beta <= 0.0:
            self.anchor = None
            return

        self.anchor = copy.deepcopy(self.policy).to(self.device).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)

    def freeze_for_stack_specialization(self, train_logstd=False):
        """
        Freeze shared representation and place-specific heads.
        Train only the stack actor head and stack critic head.

        Skill index:
          0 = place
          1 = stack
        """
        # Also freeze observation normalization.
        # Otherwise stack-heavy data changes obs_rms and can damage frozen place behavior.
        self.freeze_obs_rms = True

        # Freeze shared trunk.
        for param in self.policy.shared.parameters():
            param.requires_grad_(False)

        # Freeze place actor and place critic.
        for param in self.policy.actor_mean[0].parameters():
            param.requires_grad_(False)
        for param in self.policy.critic[0].parameters():
            param.requires_grad_(False)

        # Train stack actor and stack critic.
        for param in self.policy.actor_mean[1].parameters():
            param.requires_grad_(True)
        for param in self.policy.critic[1].parameters():
            param.requires_grad_(True)

        # Freeze logstd by default to preserve old stochastic behavior.
        self.policy.actor_logstd.requires_grad_(bool(train_logstd))

        trainable_params = [
            param for param in self.policy.parameters()
            if param.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("No trainable parameters left after freezing.")

        self.optim = torch.optim.Adam(
            trainable_params,
            lr=self.optim.param_groups[0]["lr"],
            eps=1e-5,
        )

        n_trainable = sum(param.numel() for param in trainable_params)
        n_total = sum(param.numel() for param in self.policy.parameters())

        print(
            "  >>> Stack specialization enabled: "
            "frozen shared trunk + place head + place critic; "
            "training stack head only "
            f"({n_trainable:,}/{n_total:,} params trainable)"
        )

    def select_action(self, obs_np):
        obs_norm = self.obs_rms.normalize(obs_np)
        obs = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            act, logp, val = self.policy.act(obs)

        return act.cpu().numpy(), logp.item(), val.item()

    def value(self, obs_np):
        obs_norm = self.obs_rms.normalize(obs_np)
        obs = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, _, val = self.policy(obs)

        return val.item()

    def store(self, obs, act, logp, rew, val, done):
        obs_norm = self.obs_rms.normalize(obs)

        self.buffer.add(
            torch.tensor(obs_norm, dtype=torch.float32, device=self.device),
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            torch.tensor(act, dtype=torch.float32, device=self.device),
            torch.tensor(logp, dtype=torch.float32, device=self.device),
            torch.tensor(rew, dtype=torch.float32, device=self.device),
            torch.tensor(val, dtype=torch.float32, device=self.device),
            torch.tensor(done, dtype=torch.float32, device=self.device),
        )

    def _normalize_advantages_by_skill(self, advantages):
        normalized = advantages.clone()
        skill = self.buffer.obs[..., SKILL_SLICE]
        assigned = torch.zeros_like(advantages, dtype=torch.bool)

        for skill_i in range(self.policy.n_skills):
            mask = skill[..., skill_i] > 0.5
            if not mask.any():
                continue

            values = advantages[mask]
            normalized[mask] = (values - values.mean()) / (
                values.std(unbiased=False) + 1e-8
            )
            assigned |= mask

        if not assigned.all():
            values = advantages[~assigned]
            normalized[~assigned] = (values - values.mean()) / (
                values.std(unbiased=False) + 1e-8
            )

        return normalized

    def update(self, last_obs):
        last_obs_norm = self.obs_rms.normalize(last_obs)
        last_obs_t = torch.tensor(last_obs_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, _, last_val = self.policy(last_obs_t)

        advantages, returns = self.buffer.compute_returns(last_val, self.gamma, self.lam)
        advantages = self._normalize_advantages_by_skill(advantages)

        metrics = {
            "loss_total": [],
            "loss_actor": [],
            "loss_critic": [],
            "loss_anchor": [],
            "entropy": [],
        }

        for _ in range(self.n_epochs):
            idx = torch.randperm(self.n_steps, device=self.device)

            for start in range(0, self.n_steps, self.batch_size):
                mb = idx[start : start + self.batch_size]
                obs_mb = self.buffer.obs[mb]

                new_logp, new_val, entropy = self.policy.evaluate(
                    obs_mb, self.buffer.acts[mb]
                )
                ratio = (new_logp - self.buffer.logps[mb]).exp()

                adv = advantages[mb]
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                loss_actor = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                old_val = self.buffer.vals[mb]
                val_clipped = old_val + torch.clamp(
                    new_val - old_val,
                    -self.clip_eps,
                    self.clip_eps,
                )
                loss_c1 = (new_val - returns[mb]).pow(2)
                loss_c2 = (val_clipped - returns[mb]).pow(2)
                loss_critic = 0.5 * torch.max(loss_c1, loss_c2).mean()

                loss_entropy = -entropy.mean()

                loss = (
                    loss_actor
                    + self.vf_coef * loss_critic
                    + self.ent_coef * loss_entropy
                )

                anchor_loss = torch.tensor(0.0, device=self.device)
                if self.anchor is not None and self.anchor_beta > 0.0:
                    is_place = (obs_mb[..., -3] > 0.5).float()

                    if is_place.sum() > 0:
                        with torch.no_grad():
                            anchor_mean, _, _ = self.anchor(obs_mb)

                        cur_mean, _, _ = self.policy(obs_mb)
                        per_sample = (cur_mean - anchor_mean).pow(2).mean(dim=-1)
                        anchor_loss = (per_sample * is_place).sum() / (
                            is_place.sum() + 1e-6
                        )
                        loss = loss + self.anchor_beta * anchor_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()

                metrics["loss_total"].append(loss.item())
                metrics["loss_actor"].append(loss_actor.item())
                metrics["loss_critic"].append(loss_critic.item())
                metrics["loss_anchor"].append(anchor_loss.item())
                metrics["entropy"].append(-loss_entropy.item())

        if not self.freeze_obs_rms:
            self.obs_rms.update(self.buffer.raw_obs.cpu().numpy())
        self.buffer.reset()

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path, extra=None):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "obs_rms_mean": self.obs_rms.mean,
                "obs_rms_var": self.obs_rms.var,
                "obs_rms_count": self.obs_rms.count,
                "extra": extra or {},
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["policy"]
        current = self.policy.state_dict()

        # Expand old single-head checkpoints into dual-head checkpoints.
        if "actor_mean.weight" in state:
            for i in range(self.policy.n_skills):
                state[f"actor_mean.{i}.weight"] = state["actor_mean.weight"].clone()
                state[f"actor_mean.{i}.bias"] = state["actor_mean.bias"].clone()
                state[f"critic.{i}.weight"] = state["critic.weight"].clone()
                state[f"critic.{i}.bias"] = state["critic.bias"].clone()

            for k in [
                "actor_mean.weight",
                "actor_mean.bias",
                "critic.weight",
                "critic.bias",
            ]:
                state.pop(k, None)

        # Expand old shared logstd into per-skill logstd.
        if "actor_logstd" in state and state["actor_logstd"].shape != current["actor_logstd"].shape:
            old = state["actor_logstd"]
            if old.ndim == 1 and current["actor_logstd"].ndim == 2:
                state["actor_logstd"] = old.unsqueeze(0).repeat(self.policy.n_skills, 1)

        # Handle obs-dim growth in shared trunk.
        first_key = "shared.0.weight"
        if first_key in state and state[first_key].shape != current[first_key].shape:
            old_w = state[first_key]
            new_w = torch.zeros_like(current[first_key])
            rows = min(old_w.shape[0], new_w.shape[0])
            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:rows, :cols] = old_w[:rows, :cols]
            state[first_key] = new_w

        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing:
            print(f"  [load] missing keys initialized randomly/zero: {missing}")
        if unexpected:
            print(f"  [load] unexpected keys ignored: {unexpected}")

        mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
        var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)

        if mean.shape != self.obs_rms.mean.shape:
            new_mean = np.zeros_like(self.obs_rms.mean)
            new_var = np.ones_like(self.obs_rms.var)
            n = min(mean.shape[0], new_mean.shape[0])
            new_mean[:n] = mean[:n]
            new_var[:n] = var[:n]
            mean, var = new_mean, new_var

        self.obs_rms.mean = mean
        self.obs_rms.var = var
        self.obs_rms.count = ckpt["obs_rms_count"]
        self.extra = ckpt.get("extra", {})

        self.policy.eval()
        self._make_anchor_if_enabled()

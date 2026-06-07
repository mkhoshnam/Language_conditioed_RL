import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


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
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(
            np.float32
        )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Linear(hidden, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, x):
        feat = self.shared(x)
        mean = self.actor_mean(feat)
        logstd = self.actor_logstd.clamp(-5.0, 0.0)
        std = logstd.exp().expand_as(mean)
        value = self.critic(feat).squeeze(-1)
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

        self.policy = ActorCritic(obs_dim, act_dim).to(device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim, device)
        self.obs_rms = RunningMeanStd((obs_dim,))

    def select_action(self, obs_np):
        obs_norm = self.obs_rms.normalize(obs_np)
        obs = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            act, logp, val = self.policy.act(obs)
        return act.cpu().numpy(), logp.item(), val.item()

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

    def update(self, last_obs):
        last_obs_norm = self.obs_rms.normalize(last_obs)
        last_obs_t = torch.tensor(last_obs_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, _, last_val = self.policy(last_obs_t)

        advantages, returns = self.buffer.compute_returns(last_val, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"loss_total": [], "loss_actor": [], "loss_critic": [], "entropy": []}

        for _ in range(self.n_epochs):
            idx = torch.randperm(self.n_steps, device=self.device)
            for start in range(0, self.n_steps, self.batch_size):
                mb = idx[start : start + self.batch_size]
                new_logp, new_val, entropy = self.policy.evaluate(
                    self.buffer.obs[mb], self.buffer.acts[mb]
                )
                ratio = (new_logp - self.buffer.logps[mb]).exp()

                adv = advantages[mb]
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                loss_actor = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                old_val = self.buffer.vals[mb]
                val_clipped = old_val + torch.clamp(
                    new_val - old_val, -self.clip_eps, self.clip_eps
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

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optim.step()

                metrics["loss_total"].append(loss.item())
                metrics["loss_actor"].append(loss_actor.item())
                metrics["loss_critic"].append(loss_critic.item())
                metrics["entropy"].append(-loss_entropy.item())

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
        first_key = "shared.0.weight"
        if first_key in state and state[first_key].shape != current[first_key].shape:
            old_w = state[first_key]
            new_w = current[first_key].clone()
            rows = min(old_w.shape[0], new_w.shape[0])
            cols = min(old_w.shape[1], new_w.shape[1])
            new_w[:rows, :cols] = old_w[:rows, :cols]
            if old_w.shape[1] + 1 == new_w.shape[1]:
                new_w[:rows, -1] = old_w[:rows, -1]
            state[first_key] = new_w
        self.policy.load_state_dict(state)

        mean = np.asarray(ckpt["obs_rms_mean"], dtype=np.float64)
        var = np.asarray(ckpt["obs_rms_var"], dtype=np.float64)
        if mean.shape != self.obs_rms.mean.shape:
            new_mean = np.zeros_like(self.obs_rms.mean)
            new_var = np.ones_like(self.obs_rms.var)
            n = min(mean.shape[0], new_mean.shape[0])
            new_mean[:n] = mean[:n]
            new_var[:n] = var[:n]
            if mean.shape[0] + 1 == new_mean.shape[0]:
                new_mean[-1] = mean[-1]
                new_var[-1] = var[-1]
            mean, var = new_mean, new_var
        self.obs_rms.mean = mean
        self.obs_rms.var = var
        self.obs_rms.count = ckpt["obs_rms_count"]
        self.extra = ckpt.get("extra", {})
        self.policy.eval()

"""A small PPO implementation for continuous control.

Deliberately dependency-free beyond torch: no stable-baselines, no gym. The
point of the project was to understand the algorithm, so clipping, GAE and the
value target are all written out.

Both policies in this project (`nav` and `walk`) are instances of `PPO`; they
differ only in their state/action dimensions and hidden widths.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import Actor, Critic

_mse = nn.MSELoss()

# Bounds on the learnable exploration std. See "Known issues" in the README —
# the upper bound is currently applied to the tensor that carries the gradient,
# which pins std at the ceiling rather than merely limiting it.
LOG_STD_MIN = 0.01
LOG_STD_MAX = 0.5


@dataclass
class PPOConfig:
    """Hyperparameters. The defaults are what the shipped checkpoints used."""

    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95            # GAE lambda
    clip_eps: float = 0.2
    epochs: int = 4              # optimisation passes per rollout
    batch_size: int = 512
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    init_log_std: float = -1.0   # std ~= 0.37 at start


class RolloutBuffer:
    """Experience collected from one environment between policy updates."""

    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.dones     = []

    def __len__(self):
        return len(self.rewards)

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()


class PPO:
    """Actor-critic PPO agent with a state-independent, learnable action std."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 critic_hidden_dim=512, config=None):
        self.cfg = config or PPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor  = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, critic_hidden_dim)

        # A single std per action dimension, learned rather than annealed on a
        # schedule, so the policy can keep exploring the dimensions it is still
        # unsure about and commit on the ones it has settled.
        self.log_std = nn.Parameter(
            torch.full((action_dim,), self.cfg.init_log_std)
        )

        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + [self.log_std], lr=self.cfg.lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr
        )

    # -- acting ------------------------------------------------------------

    def _std(self):
        return self.log_std.exp().clamp(LOG_STD_MIN, LOG_STD_MAX)

    @torch.no_grad()
    def get_action(self, state, deterministic=False):
        """Act on a single state. Returns (action, log_prob, value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean    = self.actor(state_t)
        dist    = torch.distributions.Normal(mean, self._std())
        action  = mean if deterministic else dist.sample()
        return (
            action.numpy().flatten(),
            dist.log_prob(action).sum(dim=-1).item(),
            self.critic(state_t).item(),
        )

    @torch.no_grad()
    def act(self, state):
        """Greedy action only — skips the critic and the distribution.

        Used by `watch.py` and `evaluate.py`, which never need a value or a
        log-prob, and call this once per 2 ms timestep.
        """
        mean = self.actor(torch.FloatTensor(state).unsqueeze(0))
        return mean.numpy().flatten()

    @torch.no_grad()
    def get_actions_batch(self, states):
        """Act on a batch of states, one per environment."""
        states_t  = torch.FloatTensor(states)
        means     = self.actor(states_t)
        dist      = torch.distributions.Normal(means, self._std())
        actions   = dist.sample()
        return (
            actions.numpy(),
            dist.log_prob(actions).sum(dim=-1).numpy(),
            self.critic(states_t).squeeze(-1).numpy(),
        )

    @torch.no_grad()
    def value_of(self, states):
        """Critic estimate for a batch of states."""
        return self.critic(torch.FloatTensor(states)).squeeze(-1).numpy()

    # -- learning ----------------------------------------------------------

    def _gae(self, buf, next_value):
        """Generalized Advantage Estimation over one buffer.

        Note: `done` conflates termination (a fall) with truncation (hitting
        the step cap), so value is not bootstrapped at a time limit. See
        "Known issues" in the README.
        """
        advantages = []
        gae        = 0
        values     = buf.values + [next_value]
        cfg        = self.cfg
        for t in reversed(range(len(buf.rewards))):
            delta = (buf.rewards[t]
                     + cfg.gamma * values[t + 1] * (1 - buf.dones[t])
                     - values[t])
            gae   = delta + cfg.gamma * cfg.lam * (1 - buf.dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def _optimise(self, states, actions, old_lp, old_vals, advantages, returns):
        cfg = self.cfg
        n   = len(states)
        for _ in range(cfg.epochs):
            indices = torch.randperm(n)
            for start in range(0, n, cfg.batch_size):
                idx      = indices[start:start + cfg.batch_size]
                s, a, lp = states[idx], actions[idx], old_lp[idx]
                adv, ret = advantages[idx], returns[idx]
                old_v    = old_vals[idx]

                dist     = torch.distributions.Normal(self.actor(s), self._std())
                new_lp   = dist.log_prob(a).sum(dim=-1)
                new_vals = self.critic(s).squeeze(-1)

                ratio      = torch.exp(new_lp - lp)
                clipped    = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                actor_loss = (-torch.min(ratio * adv, clipped * adv).mean()
                              - cfg.entropy_coef * dist.entropy().mean())

                # Clipped value loss: pessimistic, takes the worse of the raw
                # and the trust-region-limited critic error.
                val_clipped = old_v + torch.clamp(
                    new_vals - old_v, -cfg.clip_eps, cfg.clip_eps
                )
                critic_loss = torch.max(
                    _mse(new_vals, ret), _mse(val_clipped, ret)
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + [self.log_std],
                    cfg.max_grad_norm,
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )
                self.critic_optimizer.step()

    def update(self, buffers, next_values):
        """Update from N per-environment buffers.

        Each buffer gets its own GAE pass (advantages must not leak across
        environment boundaries), then every transition is pooled into one
        shuffled minibatch stream so each gradient step sees all 16 envs.
        """
        parts = ([], [], [], [], [], [])
        for buf, nv in zip(buffers, next_values):
            old_v = torch.FloatTensor(buf.values)
            adv   = torch.FloatTensor(self._gae(buf, nv))
            for dst, val in zip(parts, (
                torch.FloatTensor(np.array(buf.states)),
                torch.FloatTensor(np.array(buf.actions)),
                torch.FloatTensor(buf.log_probs),
                old_v,
                adv,
                adv + old_v,
            )):
                dst.append(val)
            buf.clear()

        states, actions, old_lp, old_vals, advantages, returns = (
            torch.cat(p) for p in parts
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self._optimise(states, actions, old_lp, old_vals, advantages, returns)

    # -- persistence -------------------------------------------------------

    def save(self, path, best_reward=None, extra=None):
        data = {
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
            "log_std": self.log_std.data,
            "actor_optimizer":  self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.cfg),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        if best_reward is not None:
            # Plain float, so the checkpoint stays loadable with weights_only=True.
            data["best_reward"] = float(best_reward)
        if extra:
            data.update(extra)
        torch.save(data, path)

    def load(self, path, load_optimizer=True):
        """Load weights. Returns the stored `best_reward`, or -inf if absent.

        Older checkpoints hold only actor/critic/log_std; the optional keys are
        each guarded so those still load.
        """
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Pre-cleanup checkpoints stored best_reward as a numpy scalar,
            # which the safe loader rejects.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "log_std" in checkpoint:
            self.log_std.data = checkpoint["log_std"]
        if load_optimizer and "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return float(checkpoint.get("best_reward", float("-inf")))

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
        return self

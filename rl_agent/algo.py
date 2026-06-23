"""
Proximal Policy Optimization (PPO) with Generalized Advantage Estimation.

This implements the clipped PPO algorithm, which is well-suited for continuous
action spaces like portfolio weight optimization. Key features:
- Clipped objective for stable policy updates
- GAE (lambda-return) for advantage estimation
- Entropy bonus for exploration
- Separate optimizers for actor and critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple

from .config import TrainingConfig, NetworkConfig, EnvironmentConfig
from .networks import ActorCritic


class Transition(NamedTuple):
    """Single environment transition stored in the rollout buffer."""

    stock_features: np.ndarray  # (n_stocks, stock_feat_dim)
    market_features: np.ndarray  # (market_feat_dim,)
    action: np.ndarray  # (n_stocks,)
    log_prob: float
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    """
    Stores transitions from environment rollouts and computes returns/advantages.
    """

    def __init__(self):
        self.transitions: list[Transition] = []

    def add(self, transition: Transition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def compute_returns_and_advantages(self,
        last_value: float, gamma: float, gae_lambda: float,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        Args:
            last_value: V(s_T+1) bootstrap value for the final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda for bias-variance tradeoff.

        Returns:
            returns: (n_steps,) discounted returns.
            advantages: (n_steps,) GAE advantages.
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        next_value = last_value

        for step in reversed(range(n)):
            tr = self.transitions[step]
            mask = 0.0 if tr.done else 1.0

            # TD error
            delta = tr.reward + gamma * next_value * mask - tr.value
            # GAE
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            
            advantages[step] = last_gae
            returns[step] = advantages[step] + tr.value

            next_value = tr.value

        return returns, advantages

    def get_batches(self,
        returns: np.ndarray, advantages: np.ndarray,
        minibatch_size: int, device: torch.device
        ):
        """
        Yield randomized minibatches from the buffer.
        """
        n = len(self.transitions)
        indices = np.random.permutation(n)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            batch_idx = indices[start:end]

            batch_transitions = [self.transitions[i] for i in batch_idx]

            yield {
                "stock_features": torch.FloatTensor(
                    np.stack([t.stock_features for t in batch_transitions])
                ).to(device),
                "market_features": torch.FloatTensor(
                    np.stack([t.market_features for t in batch_transitions])
                ).to(device),
                "actions": torch.FloatTensor(
                    np.stack([t.action for t in batch_transitions])
                ).to(device),
                "old_log_probs": torch.FloatTensor(
                    [t.log_prob for t in batch_transitions]
                ).to(device),
                "returns": torch.FloatTensor(returns[batch_idx]).to(device),
                "advantages": torch.FloatTensor(advantages[batch_idx]).to(device),
            }


class PPOAgent:
    """
    PPO Agent for portfolio optimization.
    """

    def __init__(self,
        n_stocks: int, stock_feature_dim: int, market_feature_dim: int,
        train_config: TrainingConfig | None = None,
        net_config: NetworkConfig | None = None, env_config: EnvironmentConfig | None = None
        ):
        self.config = train_config or TrainingConfig()
        self.device = torch.device(
            self.config.device
            if torch.cuda.is_available() and self.config.device == "cuda"
            else "cpu"
        )

        # Actor-Critic network
        self.ac = ActorCritic(
            n_stocks=n_stocks,
            stock_feature_dim=stock_feature_dim,
            market_feature_dim=market_feature_dim,
            net_config=net_config,
            env_config=env_config,
        ).to(self.device)

        # Separate optimizers (important for PPO stability)
        self.actor_optimizer = torch.optim.AdamW(
            self.ac.actor.parameters(), lr=self.config.lr_actor, weight_decay=1e-5,
            )
        self.critic_optimizer = torch.optim.AdamW(
            self.ac.critic.parameters(), lr=self.config.lr_critic, weight_decay=1e-5,
            )

        # Learning rate schedulers
        if self.config.use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.actor_optimizer, milestones=self.config.lr_decay_episodes,
                gamma=self.config.lr_decay_factor
                )
            self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.critic_optimizer, milestones=self.config.lr_decay_episodes,
                gamma=self.config.lr_decay_factor
                )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.train_stats: list[dict] = []

        # LSTM hidden state — maintained across steps within one episode.
        # reset_hidden_state() must be called at the start of every episode.
        self._lstm_h: tuple[torch.Tensor, torch.Tensor] | None = None

        # Per-window active-stock mask, (1, n_stocks) bool tensor or None
        self._action_mask_t: torch.Tensor | None = None

    def reset_hidden_state(self) -> None:
        """
        Reset the LSTM hidden state to zero.

        Called by:
          - Train.train()      at the start of each training episode
          - _apply_oos()       before deterministic OOS weight selection
        """
        self._lstm_h = None

    def set_action_mask(self, mask: np.ndarray | None) -> None:
        """
        Store the active-stock mask used by the critic's masked pooling.

        Called by Train.train() at the start of each window
        """
        if mask is None:
            self._action_mask_t = None
        else:
            self._action_mask_t = torch.as_tensor(
                mask, dtype=torch.bool, device=self.device
            ).view(1, -1)

    def warmup_lstm(self, market_feats_sequence) -> None:
        """
        Step the LSTM through a sequence of market feature vectors, updating
        ``self._lstm_h`` in place. No-op if LSTM is disabled.

        Used by walk-forward OOS to bring the hidden state into the same
        regime that training episodes left it in by their final step,
        instead of cold-starting from zero at every OOS prediction.
        """
        if self.ac.lstm is None:
            return
        with torch.no_grad():
            for mf in market_feats_sequence:
                mf_t = torch.FloatTensor(mf).unsqueeze(0).to(self.device)
                _, self._lstm_h = self.ac.lstm_step(mf_t, self._lstm_h)

    def select_action(self,
        stock_features: np.ndarray, market_features: np.ndarray,
        deterministic: bool = False
        ) -> tuple[np.ndarray, float, float]:
        """
        Select action given current state.

        Returns:
            action: (n_stocks,) raw action
            log_prob: scalar log probability
            value: scalar state value estimate
        """
        with torch.no_grad():
            sf = torch.FloatTensor(stock_features).unsqueeze(0).to(self.device)
            mf = torch.FloatTensor(market_features).unsqueeze(0).to(self.device)

            # Augment market features with LSTM context (no-op when LSTM disabled)
            mf_aug, self._lstm_h = self.ac.lstm_step(mf, self._lstm_h)
            action, log_prob = self.ac.actor.get_action(sf, mf_aug, deterministic, self._action_mask_t)
            value = self.ac.critic(sf, mf_aug, self._action_mask_t)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )
    
    def _update_step(self, batch: dict) -> dict:
        """Single PPO update step on a minibatch."""
        sf = batch["stock_features"]
        mf = batch["market_features"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Augment market features via LSTM (hidden=None → zero-init per sample)
        mf_aug, _ = self.ac.lstm_step(mf)

        # Current policy evaluation
        mean, std = self.ac.actor.forward(sf, mf_aug, self._action_mask_t)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        # Current value estimate
        values = self.ac.critic(sf, mf_aug, self._action_mask_t)

        # Policy loss (clipped PPO objective)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (encourages exploration)
        entropy_loss = -self.config.entropy_coeff * entropy
        policy_loss = policy_loss + entropy_loss

        # Value loss
        if self.config.critic_loss == "mse":
            value_loss = F.mse_loss(values, returns)
        else:  # huber
            value_loss = F.smooth_l1_loss(values, returns)

        # Combined backward pass — actor and critic share the LSTM graph
        total_loss = policy_loss + self.config.value_loss_coeff * value_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.ac.critic.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Diagnostics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_frac = (
                (torch.abs(ratio - 1) > self.config.clip_epsilon)
                .float().mean().item()
            )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        }

    def store_transition(self, transition: Transition):
        """Store a transition in the rollout buffer."""
        self.buffer.add(transition)

    def update(self) -> dict:
        """
        Perform PPO update using collected rollout data.

        Returns:
            Dictionary of training statistics for this update.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute bootstrap value for the last state
        last_transition = self.buffer.transitions[-1]
        if last_transition.done:
            last_value = 0.0
        else:
            with torch.no_grad():
                sf = torch.FloatTensor(last_transition.stock_features).unsqueeze(0).to(self.device)
                mf = torch.FloatTensor(last_transition.market_features).unsqueeze(0).to(self.device)
                mf_aug, _ = self.ac.lstm_step(mf)
                last_value = self.ac.critic(sf, mf_aug, self._action_mask_t).item()

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        # Normalize advantages (important for PPO stability)
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # PPO update epochs
        all_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(self.config.n_epochs_per_update):
            for batch in self.buffer.get_batches(
                returns, advantages, self.config.minibatch_size, self.device
            ):
                stats = self._update_step(batch)
                for k, v in stats.items():
                    all_stats[k].append(v)

            # KL divergence measures how much the updated policy has shifted from
            # the policy that collected the rollout data. Too large a shift means
            # the PPO importance-weight correction is no longer reliable, so we
            # stop updating early to keep the old and new policies close together.
            mean_kl = np.mean(all_stats["approx_kl"][-len(self.buffer):])
            if mean_kl > self.config.kl_target:
                break

        # Step LR schedulers
        if self.config.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        # Clear buffer
        self.buffer.clear()

        # Aggregate stats
        stats_summary = {k: float(np.mean(v)) for k, v in all_stats.items()}
        stats_summary["n_epochs"] = epoch + 1
        self.train_stats.append(stats_summary)

        return stats_summary

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "ac_state_dict": self.ac.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "train_stats": self.train_stats,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint["ac_state_dict"])
        self.ac.flatten_lstm_parameters()
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.train_stats = checkpoint.get("train_stats", [])
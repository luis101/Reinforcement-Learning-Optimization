"""
Training loop for the PPO portfolio agent.

Handles the interaction between agent and environment, data collection,
periodic evaluation, checkpointing, and logging.

Supports two modes:
- Standalone: with validation environment and full logging.
  Can be used for both initial training and final fine-tuning or for cross-validation.
- Walk-forward: silent version with early-stopping and in-memory best-checkpoint.
  Used by WalkForwardBacktestEngine._train_window via Train(verbose=False).
"""

import copy
import time
import numpy as np
from pathlib import Path

from .config import Config, TrainingConfig
from .environment import PortfolioEnv
from .algo import PPOAgent, Transition
from .utils import evaluate_agent, format_metrics


class Train:
    """
    Training loop for portfolio reinforcement learning.

    Supports both standalone training (validation env, checkpointing, full logging)
    and walk-forward mode (early stopping, in-memory best-checkpoint).
    """

    def __init__(self,
        agent: PPOAgent,
        train_env: PortfolioEnv,
        val_env: PortfolioEnv | None = None,
        config: Config | None = None,
        train_config: TrainingConfig | None = None,
        action_mask: np.ndarray | None = None,
        ):
        """
        Args:
            agent: PPO agent to train (may be pre-initialized for warm-starting).
            train_env: Environment providing the training episodes.
            val_env: Optional validation environment. If provided, periodic evaluation
                     is run and the best validation checkpoint is saved to disk.
            config: Full Config object. Ignored for training settings if train_config
                    is provided.
            train_config: TrainingConfig that takes priority over config.training.
                          Use this to pass window-specific configs from walk-forward.
            action_mask: Boolean array (n_stocks,). Inactive stocks are forced to -1e6
                         before the softmax/tanh so they receive zero weight.
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.action_mask = action_mask

        _cfg = config or Config()
        self.tc = train_config or _cfg.training
        self.ckpt_dir = Path(self.tc.checkpoint_dir)

        # Logging
        self.episode_rewards: list[float] = []
        self.eval_history: list[dict] = []
        self.best_val_sharpe: float = -np.inf

    def train(self,
              patience: int | None = None, min_episodes: int = 0, verbose: bool = True
              ) -> dict:
        """
        Run the training loop.

        Args:
            patience: Stop if the rolling-average reward (window=20) does not improve
                      by more than 0.001 for this many episodes. None disables early
                      stopping.
            min_episodes: Early stopping is not checked before this many episodes have
                          run, regardless of patience.
            verbose: Print episode progress and run a final evaluation. Set False for
                     walk-forward mode to keep output clean.

        Returns:
            dict with keys:
              episode_rewards, eval_history, final_eval, train_stats,
              episodes, best_reward, final_reward
        """
        if verbose:
            print("=" * 50)
            print("  RL Portfolio Optimization - Training")
            print("=" * 50)
            print(f"  Device:   {self.agent.device}")
            print(f"  Episodes: {self.tc.n_episodes}")
            print("=" * 50)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        best_reward = -np.inf
        best_state_dict: dict | None = None
        patience_counter = 0
        ep = 0

        for episode in range(1, self.tc.n_episodes + 1):
            ep = episode

            _, stock_feats, market_feats = self.train_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, log_prob, value = self.agent.select_action(stock_feats, market_feats)

                if self.action_mask is not None:
                    action[~self.action_mask[:len(action)]] = -1e6

                step_result = self.train_env.step(action)

                self.agent.store_transition(
                    Transition(
                        stock_features=stock_feats,
                        market_features=market_feats,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=step_result.reward,
                        done=step_result.done,
                    )
                )

                stock_feats = step_result.stock_features
                market_feats = step_result.market_features
                done = step_result.done
                episode_reward += step_result.reward

            self.episode_rewards.append(episode_reward)
            update_stats = self.agent.update()

            if verbose and (episode % 10 == 0 or episode == 1):
                elapsed = time.time() - start_time
                avg_reward = float(np.mean(self.episode_rewards[-50:]))
                print(
                    f"Episode {episode:4d}/{self.tc.n_episodes} | "
                    f"Reward: {episode_reward:+8.4f} | "
                    f"Avg (last 50): {avg_reward:+8.4f} | "
                    f"Policy loss: {update_stats.get('policy_loss', 0):+.4f} | "
                    f"Value loss: {update_stats.get('value_loss', 0):.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Periodic validation evaluation (standalone mode)
            if self.val_env is not None and episode % self.tc.eval_frequency == 0:
                result = evaluate_agent(self.agent, self.val_env)
                metrics = result["metrics"]
                self.eval_history.append({"episode": episode, "metrics": metrics})

                sharpe = metrics.get("sharpe_ratio", -np.inf)
                is_best = sharpe > self.best_val_sharpe
                if verbose:
                    print(
                        f"[Eval ep {episode}] Sharpe: {sharpe:.3f} | "
                        f"Return: {metrics.get('total_return', 0):.2%} | "
                        f"MaxDD: {metrics.get('max_drawdown', 0):.2%}"
                        + (" *BEST*" if is_best else "")
                    )
                if is_best:
                    self.best_val_sharpe = sharpe
                    self.agent.save(str(self.ckpt_dir / "agent_best.pt"))

            # In-memory best-checkpoint and early stopping
            if episode >= min_episodes:
                recent_avg = float(np.mean(self.episode_rewards[-20:]))
                if recent_avg > best_reward + 0.001:
                    best_reward = recent_avg
                    best_state_dict = copy.deepcopy(self.agent.ac.state_dict())
                    patience_counter = 0
                elif patience is not None:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at episode {episode} (patience={patience})")
                        break

            # Periodic disk checkpoint (standalone mode)
            if verbose and episode % self.tc.save_frequency == 0:
                self.agent.save(str(self.ckpt_dir / f"agent_ep{episode}.pt"))

        # Restore best weights so the returned agent has peak performance
        if best_state_dict is not None:
            self.agent.ac.load_state_dict(best_state_dict)

        # Final evaluation and disk save (standalone mode only)
        final_eval: dict = {}
        if verbose:
            print("\n" + "=" * 70)
            print("  Final Evaluation")
            print("=" * 70)
            if self.val_env:
                print("\n  Validation Set:")
                val_result = evaluate_agent(self.agent, self.val_env)
                print(format_metrics(val_result["metrics"]))
                final_eval["validation"] = val_result
            print("\n  Training Set:")
            train_result = evaluate_agent(self.agent, self.train_env)
            print(format_metrics(train_result["metrics"]))
            final_eval["training"] = train_result
            self.agent.save(str(self.ckpt_dir / "agent_final.pt"))
            print(f"\n  Final model saved to {self.ckpt_dir / 'agent_final.pt'}")

        return {
            "episode_rewards": self.episode_rewards,
            "eval_history": self.eval_history,
            "final_eval": final_eval,
            "train_stats": self.agent.train_stats,
            "episodes": ep,
            "best_reward": best_reward,
            "final_reward": float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0.0,
        }

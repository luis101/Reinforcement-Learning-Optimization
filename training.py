"""
Training loop for the PPO portfolio agent.

Handles the interaction between agent and environment, data collection,
periodic evaluation, checkpointing, and logging.
"""

import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .config import Config
from .environment import PortfolioEnv
from .algo import PPOAgent, Transition
from .utils import (
    compute_portfolio_metrics,
    evaluate_agent,
    format_metrics,
    compute_benchmark_returns,
)


class Train:
    """
    Training loop for portfolio reinforcement learning
    """

    def __init__(self,
        agent: PPOAgent,
        train_env: PortfolioEnv, val_env: PortfolioEnv | None = None,
        config: Config | None = None,
        ):
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.config = config or Config()
        self.tc = self.config.training

        # Logging
        self.episode_rewards: list[float] = []
        self.episode_metrics: list[dict] = []
        self.eval_history: list[dict] = []
        self.best_val_sharpe: float = -np.inf

        # Checkpoint directory
        self.ckpt_dir = Path(self.tc.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history and final evaluation results.
        """
        print("=" * 50)
        print("  RL Portfolio Optimization - Training")
        print("=" * 50)
        print(f"  Device:           {self.agent.device}")
        print(f"  Episodes:         {self.tc.n_episodes}")
        print("=" * 50)

        start_time = time.time()

        for episode in range(1, self.tc.n_episodes + 1):

            # Collect one episode of experience.
            flat_state, stock_feats, market_feats = self.train_env.reset()
            episode_reward = 0.0
            done = False
            all_daily_returns = []

            while not done:
                # Select action
                action, log_prob, value = self.agent.select_action(
                    stock_feats, market_feats
                )

                # Environment step
                step_result = self.train_env.step(action)

                # Store transition
                self.agent.store_transition(
                    Transition(
                        stock_features=stock_feats, market_features=market_feats,
                        action=action, log_prob=log_prob, value=value,
                        reward=step_result.reward, done=step_result.done,
                    )
                )

                # Update state
                stock_feats = step_result.stock_features
                market_feats = step_result.market_features
                done = step_result.done
                episode_reward += step_result.reward

                if "daily_returns" in step_result.info:
                    all_daily_returns.extend(step_result.info["daily_returns"].tolist())

            episode_info = {
                "total_reward": episode_reward,
                "portfolio_value": self.train_env.portfolio_value_series[-1],
                "n_trades": len(self.train_env._trade_history),
            }

            # Store episode stats
            self.episode_rewards.append(episode_reward)

            # PPO update
            update_stats = self.agent.update()

            # Logging
            if episode % 10 == 0 or episode == 1:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(
                    f"Episode {episode:4d}/{self.tc.n_episodes} | "
                    f"Reward: {episode_reward:+8.4f} | "
                    f"Average Reward (last 50): {avg_reward:+8.4f} | "
                    f"Policy_loss: {update_stats.get('policy_loss', 0):+.4f} | "
                    f"Value_loss: {update_stats.get('value_loss', 0):.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Periodic evaluation
            if episode % self.tc.eval_frequency == 0:
                # Evaluate on validation set and log results
                if self.val_env is None:
                    return

                result = evaluate_agent(self.agent, self.val_env)
                metrics = result["metrics"]

                self.eval_history.append(
                    {"episode": episode, "metrics": metrics}
                )

                sharpe = metrics.get("sharpe_ratio", -np.inf)
                is_best = sharpe > self.best_val_sharpe

                print(
                    f"  [Eval ep {episode}] "
                    f"Sharpe: {sharpe:.3f} | "
                    f"Return: {metrics.get('total_return', 0):.2%} | "
                    f"MaxDD: {metrics.get('max_drawdown', 0):.2%}"
                    + (" *BEST*" if is_best else "")
                )

                if is_best:
                    self.best_val_sharpe = sharpe
                    self.agent.save(str(self.ckpt_dir / "agent_best.pt"))

            # Checkpointing
            if episode % self.tc.save_frequency == 0:
                self.agent.save(str(self.ckpt_dir / f"agent_ep{episode}.pt"))




        # Final evaluation
        print("\n" + "=" * 70)
        print("  Final Evaluation")
        print("=" * 70)

        final_eval = {}
        if self.val_env:
            print("\n  Validation Set:")
            val_result = evaluate_agent(self.agent, self.val_env)
            print(format_metrics(val_result["metrics"]))
            final_eval["validation"] = val_result

        print("\n  Training Set:")
        train_result = evaluate_agent(self.agent, self.train_env)
        print(format_metrics(train_result["metrics"]))
        final_eval["training"] = train_result

        # Save final model
        self.agent.save(str(self.ckpt_dir / "agent_final.pt"))
        print(f"\n  Final model saved to {self.ckpt_dir / 'agent_final.pt'}")

        return {
            "episode_rewards": self.episode_rewards,
            "eval_history": self.eval_history,
            "final_eval": final_eval,
            "train_stats": self.agent.train_stats,
        }


def create_and_train(
    prices: pd.DataFrame, config: Config | None = None
    ) -> tuple[PPOAgent, dict]:
    """
    Utility function to create all components and train

    Args:
        prices: DataFrame (n_days, n_stocks) of adjusted close prices.
        config: Master configuration. Uses defaults if None.

    Returns:
        (trained_agent, training_results)
    """
    config = config or Config()

    # Set seed
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    # Split data
    n = len(prices)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    warmup = config.features.normalize_window + 63

    train_prices = prices.iloc[:train_end]
    val_prices = prices.iloc[max(0, train_end - warmup) : val_end]

    # Create environments
    train_env = PortfolioEnv(train_prices, config.env, config.features)
    val_env = PortfolioEnv(val_prices, config.env, config.features)

    # Create agent
    agent = PPOAgent(
        n_stocks=prices.shape[1],
        stock_feature_dim=train_env.stock_feature_dim,
        market_feature_dim=train_env.market_feature_dim,
        train_config=config.training,
        net_config=config.network,
        env_config=config.env,
    )

    param_count = sum(p.numel() for p in agent.ac.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Train
    training = Train(agent, train_env, val_env, config)
    results = training.train()

    return agent, results
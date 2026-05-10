"""
Neural network architectures for the Actor-Critic agent

- Cross-asset attention layer to capture between stock relationships (correlation, hedging, etc.) 
- Separate actor and critic heads (more stable for PPO)
- Action parameterization accounts for portfolio constraints
- Layer normalization + dropout for regularization as options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# from typing import Literal

from .config import NetworkConfig, EnvironmentConfig


class CrossAssetAttention(nn.Module):
    """
    Multi-head attention over stocks to capture cross-asset dependencies.

    Each stock has a feature vector. Attention allows the model to learn
    which stocks' features are relevant for deciding each stock's weight.
    
    Attention yields a new representation for each stock that incorporates information 
    from the whole market. For example, if stock A is a strong hedge against stock B, 
    the attention can learn to give more weight to A's features when deciding B's weight.
    """

    def __init__(self, 
                 input_dim: int, attention_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.1
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = attention_dim // n_heads
        assert attention_dim % n_heads == 0

        self.q_proj = nn.Linear(input_dim, attention_dim)
        self.k_proj = nn.Linear(input_dim, attention_dim)
        self.v_proj = nn.Linear(input_dim, attention_dim)
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        self.norm = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_stocks, input_dim) per-stock features.

        Returns:
            (batch, n_stocks, attention_dim) attention-processed features.
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        out = self.norm(out)

        return out


class MLP(nn.Module):
    """
    Standard multi-layer perceptron with optional layer norm and dropout.
    Used for both actor and critic heads after attention processing.
    """

    def __init__(self,
        input_dim: int, hidden_dims: list[int], output_dim: int,
        activation: str = "gelu", dropout: float = 0.1,
        use_layer_norm: bool = True, output_activation: nn.Module | None = None
        ):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims
        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation]

        # Build hidden layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNetwork(nn.Module):
    """
    Policy network that outputs portfolio weight parameters.

    Architecture:
    1. Cross-asset attention on per-stock features, results in compressed stock embeddings
    2. Concatenate market features and flattened stock embeddings
    3. MLP to produce action mean (and optionally log_std)
    4. Action processing layer enforces portfolio constraints

    The actor outputs a Gaussian distribution over raw actions, which are then
    mapped to valid portfolio weights via softmax (long-only) or
    tanh + centering (long-short).
    """

    def __init__(self,
        n_stocks: int, stock_feature_dim: int, market_feature_dim: int, 
        net_config: NetworkConfig, env_config: EnvironmentConfig, activation: str = "gelu"
        ):
        super().__init__()
        self.n_stocks = n_stocks
        self.env_config = env_config

        # Cross-asset attention
        self.use_attention = net_config.use_attention
        if self.use_attention:
            self.attention = CrossAssetAttention(
                input_dim=stock_feature_dim,
                attention_dim=net_config.attention_dim,
                n_heads=net_config.attention_heads,
                dropout=net_config.dropout,
            )
            stock_embed_dim = net_config.attention_dim
        else:
            stock_embed_dim = stock_feature_dim

        # Stock embedding aggregation
        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation]
        self.stock_compress = nn.Sequential(
            nn.Linear(stock_embed_dim, 32),
            act_fn,
        )

        # Combine compressed stock features + market features
        combined_dim = n_stocks * 32 + market_feature_dim

        # Policy head
        self.policy_mlp = MLP(
            input_dim=combined_dim,
            hidden_dims=net_config.actor_hidden_dims,
            output_dim=n_stocks,
            activation=net_config.activation,
            dropout=net_config.dropout,
            use_layer_norm=net_config.use_layer_norm,
        )

        # Initialize output layer according to config.
        # All three options keep weights small so softmax ≈ equal-weight at start
        # while ensuring non-zero W_out so gradients flow to hidden layers.
        out_layer = self.policy_mlp.net[-1]
        init = net_config.policy_output_init
        if init == "orthogonal":
            nn.init.orthogonal_(out_layer.weight, gain=0.01)
        elif init == "normal":
            nn.init.normal_(out_layer.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(n_stocks))
        self.log_std_min = net_config.actor_log_std_min
        self.log_std_max = net_config.actor_log_std_max

    def forward(self, 
                stock_features: torch.Tensor, market_features: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stock_features: (batch, n_stocks, stock_feature_dim)
            market_features: (batch, market_feature_dim)

        Returns:
            action_mean: (batch, n_stocks) raw action means
            action_std: (batch, n_stocks) action standard deviations
        """
        B = stock_features.shape[0]

        # Cross-asset attention
        if self.use_attention:
            stock_embed = self.attention(stock_features)  # (B, N, attn_dim)
        else:
            stock_embed = stock_features

        # Compress per-stock embeddings
        stock_compressed = self.stock_compress(stock_embed)  # (B, N, 32)
        stock_flat = stock_compressed.view(B, -1)  # (B, N*32)
        # Combine with market features
        combined = torch.cat([stock_flat, market_features], dim=-1)

        # Policy output
        action_mean = self.policy_mlp(combined)  # (B, N)

        # Clamped log_std → std
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        # action_std = log_std.exp().expand_as(action_mean)
        action_std = torch.exp(log_std)

        return action_mean, action_std

    def get_action(self, 
        stock_features: torch.Tensor, market_features: torch.Tensor,
        deterministic: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute its log probability.

        Returns:
            action: (batch, n_stocks) raw actions
            log_prob: (batch,) log probability of the action
        """
        # Get the action distribution
        mean, std = self.forward(stock_features, market_features)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        # Sum log probs across stocks for the joint action log prob
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Value network that estimates the state value V(s).

    Uses the same attention mechanism as the actor but with its own parameters
    to maintain independence.
    """

    def __init__(self,
        n_stocks: int, stock_feature_dim: int, market_feature_dim: int,
        net_config: NetworkConfig, activation: str = "gelu"
        ):
        super().__init__()

        self.use_attention = net_config.use_attention
        if self.use_attention:
            self.attention = CrossAssetAttention(
                input_dim=stock_feature_dim,
                attention_dim=net_config.attention_dim,
                n_heads=net_config.attention_heads,
                dropout=net_config.dropout,
            )
            stock_embed_dim = net_config.attention_dim
        else:
            stock_embed_dim = stock_feature_dim

        # Stock embedding aggregation
        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation]
        self.stock_compress = nn.Sequential(
            nn.Linear(stock_embed_dim, 32),
            act_fn,
        )

        combined_dim = n_stocks * 32 + market_feature_dim

        self.value_mlp = MLP(
            input_dim=combined_dim,
            hidden_dims=net_config.critic_hidden_dims,
            output_dim=1,
            activation=net_config.activation,
            dropout=net_config.dropout,
            use_layer_norm=net_config.use_layer_norm,
        )

    def forward(self, 
                stock_features: torch.Tensor, market_features: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            stock_features: (batch, n_stocks, stock_feature_dim)
            market_features: (batch, market_feature_dim)

        Returns:
            value: (batch,) estimated state value
        """
        B = stock_features.shape[0]

        if self.use_attention:
            stock_embed = self.attention(stock_features)
        else:
            stock_embed = stock_features

        stock_compressed = self.stock_compress(stock_embed)
        stock_flat = stock_compressed.view(B, -1)

        combined = torch.cat([stock_flat, market_features], dim=-1)
        value = self.value_mlp(combined).squeeze(-1)

        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module.

    Wraps the actor and critic networks together, both have fully separate parameters.
    """

    def __init__(self,
        n_stocks: int, stock_feature_dim: int, market_feature_dim: int,
        net_config: NetworkConfig | None = None, env_config: EnvironmentConfig | None = None
        ):
        super().__init__()
        net_config = net_config or NetworkConfig()
        env_config = env_config or EnvironmentConfig()

        self.actor = ActorNetwork(
            n_stocks, stock_feature_dim, market_feature_dim,
            net_config, env_config,
        )
        self.critic = CriticNetwork(
            n_stocks, stock_feature_dim, market_feature_dim,
            net_config,
        )

    def forward(self,
                stock_features: torch.Tensor, market_features: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_mean, action_std, value
        """
        mean, std = self.actor(stock_features, market_features)
        value = self.critic(stock_features, market_features)

        return mean, std, value
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_stocks, input_dim) per-stock features.
            mask: optional (1, n_stocks) or (batch, n_stocks) tensor, mask for active stocks

        Returns:
            (batch, n_stocks, attention_dim) attention-processed features.
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, N_query, N_key)
        if mask is not None:
            # Key mask: Not attending to inactive assets (B or 1, 1, 1, N_key)
            key_mask = mask.reshape(-1, 1, 1, N).bool()
            attn = attn.masked_fill(~key_mask, torch.finfo(attn.dtype).min)
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
        self.policy_head_type = net_config.policy_head
        self.use_two_stage = net_config.use_two_stage

        if self.policy_head_type == "flatten_mlp":
            # Original: compress per-stock to 32, flatten, MLP -> n_stocks logits.
            # Both first hidden layer and output layer are position-specific.
            self.stock_compress = nn.Sequential(
                nn.Linear(stock_embed_dim, 32),
                act_fn,
            )
            policy_input_dim = n_stocks * 32 + market_feature_dim
            policy_output_dim = n_stocks
        elif self.policy_head_type == "shared_head_compressed":
            # Compress per-stock to 32, then apply a shared per-stock MLP to
            # each stock's (32 + market) vector. Permutation-equivariant.
            self.stock_compress = nn.Sequential(
                nn.Linear(stock_embed_dim, 32),
                act_fn,
            )
            policy_input_dim = 32 + market_feature_dim
            policy_output_dim = 1
        elif self.policy_head_type == "shared_head":
            # No compression — apply a shared per-stock MLP directly on the full
            # attention output + market features per stock.
            self.stock_compress = None
            policy_input_dim = stock_embed_dim + market_feature_dim
            policy_output_dim = 1
        else:
            raise ValueError(f"Unknown policy_head: {self.policy_head_type}")

        self.policy_mlp = MLP(
            input_dim=policy_input_dim,
            hidden_dims=net_config.actor_hidden_dims,
            output_dim=policy_output_dim,
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
        elif init == "kaiming_uniform":
            nn.init.kaiming_uniform_(out_layer.weight, mode="fan_in", nonlinearity="relu")
        elif init == "kaiming_normal":
            nn.init.kaiming_normal_(out_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(out_layer.bias)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(n_stocks))
        self.log_std_min = net_config.actor_log_std_min
        self.log_std_max = net_config.actor_log_std_max

        # Optional two-stage inclusion gate: log_sigmoid(gate_logit) is added to
        # action logits before softmax, softly zeroing low-conviction stocks
        # without forcing hard full-position entry or exit.
        if self.use_two_stage:
            if self.policy_head_type == "flatten_mlp":
                self.inclusion_head = nn.Linear(policy_input_dim, n_stocks)
            else:
                self.inclusion_head = nn.Linear(policy_input_dim, 1)
            nn.init.orthogonal_(self.inclusion_head.weight, gain=0.01)
            nn.init.zeros_(self.inclusion_head.bias)

        # Learnable softmax temperature: concentrates weights as it decreases.
        # Initialised at temperature_init (≈1 → near-uniform) and learned from data.
        self.use_temperature = net_config.use_temperature
        if self.use_temperature:
            self.temperature = nn.Parameter(
                torch.full((1,), net_config.temperature_init)
            )
            self.temperature_min = net_config.temperature_min

    def forward(self,
                stock_features: torch.Tensor, market_features: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stock_features: (batch, n_stocks, stock_feature_dim)
            market_features: (batch, market_feature_dim)
            mask: optional (1, n_stocks) tensor, mask for active stocks

        Returns:
            action_mean: (batch, n_stocks) raw action means
            action_std: (batch, n_stocks) action standard deviations
        """
        B = stock_features.shape[0]

        # Cross-asset attention
        if self.use_attention:
            stock_embed = self.attention(stock_features, mask)  # (B, N, attn_dim)
        else:
            stock_embed = stock_features

        if self.policy_head_type == "flatten_mlp":
            # Compress per-stock embeddings, flatten, append market features, then
            # MLP -> n_stocks logits (position-specific).
            stock_compressed = self.stock_compress(stock_embed)  # (B, N, 32)
            stock_flat = stock_compressed.view(B, -1)            # (B, N*32)
            policy_input = torch.cat([stock_flat, market_features], dim=-1)
            action_mean = self.policy_mlp(policy_input)          # (B, N)
            if self.use_two_stage:
                action_mean = action_mean + F.logsigmoid(self.inclusion_head(policy_input))
        else:
            # Shared per-stock head: apply the same MLP to every stock's
            # (embedding + broadcast market features) vector and read off one logit each.
            if self.stock_compress is not None:
                stock_embed = self.stock_compress(stock_embed)   # (B, N, 32)
            N = stock_embed.shape[1]
            mkt_exp = market_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, M)
            policy_input = torch.cat([stock_embed, mkt_exp], dim=-1)  # (B, N, embed+M)
            action_mean = self.policy_mlp(policy_input).squeeze(-1)   # (B, N)
            # Two-stage gate: adds log P(include | state) to logits before softmax.
            # Stocks the gate assigns low probability to get subtracted from their
            # logit, driving their softmax weight toward zero without hard selection.
            if self.use_two_stage:
                gate = self.inclusion_head(policy_input).squeeze(-1)  # (B, N)
                action_mean = action_mean + F.logsigmoid(gate)

        # Temperature scaling: divides logits to sharpen the softmax distribution.
        # A lower temperature concentrates weight on fewer high-conviction stocks.
        if self.use_temperature:
            action_mean = action_mean / self.temperature.clamp(min=self.temperature_min)

        # Clamped log_std → std
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        # action_std = log_std.exp().expand_as(action_mean)
        action_std = torch.exp(log_std)

        return action_mean, action_std

    def get_action(self,
        stock_features: torch.Tensor, market_features: torch.Tensor,
        deterministic: bool = False, mask: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute its log probability.

        Returns:
            action: (batch, n_stocks) raw actions
            log_prob: (batch,) log probability of the action
        """
        # Get the action distribution
        mean, std = self.forward(stock_features, market_features, mask)
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
        
        # Permutation-invariant aggregation: mean + max pool over the stock axis.
        # V(s) must not depend on stock ordering (or count), so we pool with symmetric
        # functions instead of flattening N*32, which tied the value to slot position.
        #combined_dim = n_stocks * 32 + market_feature_dim
        pooled_dim = 32 * 2  # mean concat max
        combined_dim = pooled_dim + market_feature_dim

        self.value_mlp = MLP(
            input_dim=combined_dim,
            hidden_dims=net_config.critic_hidden_dims,
            output_dim=1,
            activation=net_config.activation,
            dropout=net_config.dropout,
            use_layer_norm=net_config.use_layer_norm,
        )

    def forward(self,
                stock_features: torch.Tensor, market_features: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        Args:
            stock_features: (batch, n_stocks, stock_feature_dim)
            market_features: (batch, market_feature_dim)
            mask: optional (1, n_stocks) tensor, mask for active stocks

        Returns:
            value: (batch,) estimated state value
        """
        B, N, _ = stock_features.shape

        if self.use_attention:
            stock_embed = self.attention(stock_features, mask)
        else:
            stock_embed = stock_features

        stock_compressed = self.stock_compress(stock_embed) # (B, N, 32)
        #stock_flat = stock_compressed.view(B, -1)
        if mask is None:
            pooled = torch.cat(
                [stock_compressed.mean(dim=1), stock_compressed.amax(dim=1)],
                dim=-1,
            ) # (B, 64) - order-independent
        else:
            # Masked mean + max over active stocks only, m: (1, N, 1)
            m = mask.reshape(1, N, 1).to(stock_compressed.dtype)
            masked_mean = (stock_compressed * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            neg_inf = torch.finfo(stock_compressed.dtype).min
            masked_max = stock_compressed.masked_fill(m == 0, neg_inf).amax(dim=1)
            pooled = torch.cat([masked_mean, masked_max], dim=-1) # (B, 64)

        #combined = torch.cat([stock_flat, market_features], dim=-1)
        combined = torch.cat([pooled, market_features], dim=-1)
        value = self.value_mlp(combined).squeeze(-1)

        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module.

    Wraps the actor and critic networks together, both have fully separate parameters.
    An optional shared LSTM augments market features with temporal context carried
    across rebalancing steps within each episode.
    """

    def __init__(self,
        n_stocks: int, stock_feature_dim: int, market_feature_dim: int,
        net_config: NetworkConfig | None = None, env_config: EnvironmentConfig | None = None
        ):
        super().__init__()
        net_config = net_config or NetworkConfig()
        env_config = env_config or EnvironmentConfig()

        # Optional LSTM: processes market features sequentially within an episode.
        # Its output is concatenated to the raw market features, increasing the
        # effective market_feature_dim seen by actor and critic.
        self.lstm: nn.LSTM | None = None
        actor_market_dim = market_feature_dim
        if net_config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=market_feature_dim,
                hidden_size=net_config.lstm_hidden_dim,
                batch_first=True,
            )
            actor_market_dim = market_feature_dim + net_config.lstm_hidden_dim

        self.actor = ActorNetwork(
            n_stocks, stock_feature_dim, actor_market_dim,
            net_config, env_config,
        )
        self.critic = CriticNetwork(
            n_stocks, stock_feature_dim, actor_market_dim,
            net_config,
        )

    def lstm_step(
        self,
        market_features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Augment market features with LSTM context.

        Args:
            market_features: (batch, market_feature_dim)
            hidden: LSTM (h, c) state from the previous step, or None.

        Returns:
            augmented_market_features: (batch, market_feature_dim + lstm_hidden_dim)
                                        or the original tensor when LSTM is disabled.
            new_hidden: updated (h, c) tuple, or None when LSTM is disabled.
        """
        if self.lstm is None:
            return market_features, None
        self.lstm.flatten_parameters()
        lstm_out, new_hidden = self.lstm(market_features.unsqueeze(1), hidden)
        return torch.cat([market_features, lstm_out.squeeze(1)], dim=-1), new_hidden

    def flatten_lstm_parameters(self) -> None:
        """Call after load_state_dict or deepcopy to restore cuDNN-required contiguity."""
        if self.lstm is not None:
            self.lstm.flatten_parameters()

    def forward(self,
                stock_features: torch.Tensor, market_features: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Returns:
            action_mean, action_std, value, new_hidden
        """
        mf_aug, new_hidden = self.lstm_step(market_features, hidden)
        mean, std = self.actor(stock_features, mf_aug, mask)
        value = self.critic(stock_features, mf_aug, mask)
        return mean, std, value, new_hidden
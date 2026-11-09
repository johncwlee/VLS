"""Networks for the π0.5 velocity residual trained with QAM-E: residual MLP, critic ensemble, edit actor.

Layer conventions follow the QAM reference (modules/qam/utils/networks.py): Dense -> activation -> LayerNorm,
xavier-uniform kernels, zero biases.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def mlp(in_dim: int, hidden_dims: Sequence[int], out_dim: int | None, activation: str, layer_norm: bool) -> nn.Sequential:
    """Hidden layers are Dense -> act -> (LN); an optional final Dense has no activation."""
    layers, dim = [], in_dim
    for width in hidden_dims:
        layers += [nn.Linear(dim, width), ACTIVATIONS[activation]()]
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        dim = width
    if out_dim is not None:
        layers.append(nn.Linear(dim, out_dim))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


def sinusoidal(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """[B] flow times in [0, 1] -> [B, dim] sin/cos features."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def cap(raw: Tensor, v_vla: Tensor, c: float | None, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Soft per-token cap ||v_phi_i|| <= c * ||v_vla_i|| (spec §3.4). Returns (capped, saturated-token mask)."""
    if c is None:
        return raw, torch.zeros(raw.shape[:-1], dtype=torch.bool, device=raw.device)
    limit = c * v_vla.norm(dim=-1, keepdim=True) + eps
    n_raw = raw.norm(dim=-1, keepdim=True) + eps
    return raw * (limit * torch.tanh(n_raw / limit) / n_raw), (n_raw > limit).squeeze(-1)


class ResidualMLP(nn.Module):
    """v0 velocity residual (spec §3.2, without keypoints). Output is zero at init, so π0.5 + v_phi == π0.5."""

    def __init__(self, horizon: int, action_dim: int, ae_dim: int, prefix_dim: int, state_dim: int, cfg):
        super().__init__()
        self.horizon, self.action_dim, self.cap_c, self.time_emb_dim = horizon, action_dim, cfg.cap_c, cfg.time_emb_dim
        self.prefix_proj = nn.Linear(prefix_dim, cfg.prefix_proj_dim)
        in_dim = 2 * horizon * action_dim + cfg.time_emb_dim + ae_dim + cfg.prefix_proj_dim + state_dim
        self.trunk = mlp(in_dim, cfg.hidden_dims, None, cfg.activation, cfg.layer_norm)
        self.out = nn.Linear(cfg.hidden_dims[-1], horizon * action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor, v_vla: Tensor, ae_pool: Tensor, h_prefix: Tensor, state: Tensor
                ) -> tuple[Tensor, Tensor]:
        """x, v_vla: [B, H, A] real dims; t: [B] (openpi time); ae_pool: [B, D_ae]. Returns (v_phi, saturated)."""
        inp = torch.cat([x.flatten(1), v_vla.flatten(1), sinusoidal(t, self.time_emb_dim), ae_pool,
                         self.prefix_proj(h_prefix), state], dim=-1)
        raw = self.out(self.trunk(inp)).view(-1, self.horizon, self.action_dim)
        return cap(raw, v_vla, self.cap_c)


class CriticEnsemble(nn.Module):
    """Q(s, a) ensemble; returns [E, B]."""

    def __init__(self, in_dim: int, cfg):
        super().__init__()
        self.members = nn.ModuleList(
            mlp(in_dim, cfg.hidden_dims, 1, cfg.activation, cfg.layer_norm) for _ in range(cfg.num_qs))

    def forward(self, feat: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([feat, action.flatten(1)], dim=-1)
        return torch.stack([m(x).squeeze(-1) for m in self.members])


class EditActor(nn.Module):
    """QAM-E edit policy: TanhNormal over the flattened executed chunk, conditioned on (obs features, flow chunk)."""

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0  # QAM's Normal head

    def __init__(self, feat_dim: int, edit_dim: int, cfg):
        super().__init__()
        self.trunk = mlp(feat_dim + edit_dim, cfg.hidden_dims, None, cfg.activation, layer_norm=False)
        self.mean = nn.Linear(cfg.hidden_dims[-1], edit_dim)
        self.log_std = nn.Linear(cfg.hidden_dims[-1], edit_dim)
        for layer in (self.mean, self.log_std):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.init_alpha)))

    def forward(self, feat: Tensor, chunk: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (edit in [-1, 1] with chunk's shape, log-prob [B]) for a reparameterized sample."""
        h = self.trunk(torch.cat([feat, chunk.flatten(1)], dim=-1))
        mean = self.mean(h)
        std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        u = mean + std * torch.randn_like(mean)
        log_prob = torch.distributions.Normal(mean, std).log_prob(u).sum(-1)
        log_prob = log_prob - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)  # tanh Jacobian
        return torch.tanh(u).view_as(chunk), log_prob

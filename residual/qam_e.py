"""QAM-E: Q-learning with Adjoint Matching plus an edit policy, for a velocity residual on a frozen flow policy.

PyTorch port of RL2-VLA/third_party/qam/agents/qam.py (same algorithm as modules/qam/agents/qam.py) in its
`residual=True`, `edit_scale > 0` configuration: the frozen base flow (π0.5) is QAM's `actor_slow`, the
residual v_phi is `actor_fast`, and the edit policy refines the executed chunk to maximize Q.

Flow-time conventions (spec §2.1):
    openpi / lerobot π0.5: x_t = t*noise + (1-t)*a, v = noise - a, sampling t: 1 -> 0.
    QAM:                   x_s = (1-s)*noise + s*a, v = a - noise, sampling s: 0 -> 1.
    s = 1 - t and v_q(x, s) = -v_o(x, 1 - s).
The residual is defined and called in openpi convention; the QAM math below runs in QAM convention.
"""

import contextlib
import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from residual.networks import CriticEnsemble, EditActor, ResidualMLP

# (base state, x [B, H, D], t [B]) -> (velocity [B, H, D], action-expert hidden [B, H, D_ae]), openpi convention
BaseFn = Callable[[Any, Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass
class Context:
    """Per-observation inputs. `base` is opaque state for the base flow (π0.5's prefix KV cache and masks)."""
    h_prefix: Tensor  # [B, D_p] pooled prefix features under the original instruction
    state: Tensor     # [B, S] normalized proprioception
    progress: Tensor  # [B] chunk index / max episode chunks
    base: Any


@dataclass(frozen=True)
class Dims:
    horizon: int       # H, predicted chunk length
    pad_dim: int       # D, padded action dim of the flow state
    action_dim: int    # A, real action dims
    exec_horizon: int  # actions executed per chunk (critic / edit act on [:exec_horizon, :A])
    ae_dim: int        # action-expert width
    prefix_dim: int
    state_dim: int
    flow_steps: int


@contextlib.contextmanager
def frozen(module: nn.Module):
    """Use a module inside a loss without accumulating gradients into it."""
    flags = [p.requires_grad for p in module.parameters()]
    module.requires_grad_(False)
    try:
        yield module
    finally:
        for p, flag in zip(module.parameters(), flags, strict=True):
            p.requires_grad_(flag)


class QAMEAgent(nn.Module):
    def __init__(self, cfg, dims: Dims, base_fn: BaseFn, action_low: Tensor, action_high: Tensor):
        super().__init__()
        self.cfg, self.dims, self.base_fn = cfg, dims, base_fn
        self.register_buffer("action_low", action_low.float())  # env bounds in normalized action space
        self.register_buffer("action_high", action_high.float())
        exec_dim = dims.exec_horizon * dims.action_dim
        self.feat_dim = dims.prefix_dim + dims.state_dim + int(cfg.critic.time_feature)
        self.residual = ResidualMLP(dims.horizon, dims.action_dim, dims.ae_dim, dims.prefix_dim, dims.state_dim,
                                    cfg.residual)
        self.critic = CriticEnsemble(self.feat_dim + exec_dim, cfg.critic)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.edit = EditActor(self.feat_dim, exec_dim, cfg.edit) if cfg.edit.scale > 0 else None
        self.target_entropy = -cfg.edit.target_entropy_multiplier * exec_dim
        groups = {"residual": self.residual.parameters(), "critic": self.critic.parameters()}
        if self.edit is not None:
            groups["edit"] = [p for n, p in self.edit.named_parameters() if n != "log_alpha"]
            groups["alpha"] = [self.edit.log_alpha]
        self.optims = {k: torch.optim.Adam(list(v), lr=cfg.qam.lr) for k, v in groups.items()}
        self.timer = lambda name: contextlib.nullcontext()  # the trainer swaps in monitor.Timer for update/time/*

    # ---------------------------------------------------------------- helpers
    def feat(self, ctx: Context) -> Tensor:
        parts = [ctx.h_prefix, ctx.state] + ([ctx.progress[:, None]] if self.cfg.critic.time_feature else [])
        return torch.cat(parts, dim=-1)

    def clip(self, a: Tensor) -> Tensor:
        return torch.clamp(a, self.action_low, self.action_high) if self.cfg.qam.clip_actions else a

    def exec_slice(self, x: Tensor) -> Tensor:
        return x[:, : self.dims.exec_horizon, : self.dims.action_dim]

    def _residual(self, x: Tensor, t: Tensor, v_vla: Tensor, ae_pool: Tensor, ctx: Context) -> tuple[Tensor, Tensor]:
        a = self.dims.action_dim
        return self.residual(x[..., :a], t, v_vla[..., :a], ae_pool, ctx.h_prefix, ctx.state)

    def _pad(self, v_phi: Tensor) -> Tensor:
        return F.pad(v_phi, (0, self.dims.pad_dim - self.dims.action_dim))

    def pessimistic_q(self, ctx: Context, action: Tensor, target: bool = False) -> Tensor:
        qs = (self.target_critic if target else self.critic)(self.feat(ctx), action)
        return qs.mean(0) - self.cfg.qam.rho * qs.std(0, unbiased=False)

    # ---------------------------------------------------------------- sampling (openpi convention)
    @torch.no_grad()
    def sample_chunk(self, ctx: Context, noise: Tensor, use_residual: bool) -> tuple[Tensor, dict]:
        """Euler ODE t: 1 -> 0 exactly as lerobot's PI05Pytorch.sample_actions, plus v_phi when use_residual."""
        steps, bsize = self.dims.flow_steps, noise.shape[0]
        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        x, latent, ratios, sats = noise, None, [], []
        while time >= -dt / 2:
            t = time.expand(bsize)
            v, hidden = self.base_fn(ctx.base, x, t)
            ae_pool = hidden.mean(1)
            latent = ae_pool if latent is None else latent  # first flow step: RL²-VLA / SAFE latent
            if use_residual:
                v_phi, sat = self._residual(x, t, v, ae_pool, ctx)
                ratios.append(v_phi.norm(dim=-1).mean() / (v[..., : self.dims.action_dim].norm(dim=-1).mean() + 1e-8))
                sats.append(sat.float().mean())
                v = v + self._pad(v_phi)
            x = x + dt * v
            time = time + dt
        stats = {"latent": latent}
        if use_residual:
            stats.update(ratio=torch.stack(ratios).mean(), sat=torch.stack(sats).mean())
        return x, stats

    @torch.no_grad()
    def act(self, ctx: Context, noise: Tensor, use_residual: bool = True, use_edit: bool = True,
            with_base: bool = False) -> dict:
        """Executed chunk [B, n_exec, A] (normalized, clipped), plus diagnostics and the same-noise base chunk."""
        use_residual = use_residual and self.cfg.qam.inv_temp > 0  # QAM: inv_temp == 0 samples the base only
        x0, stats = self.sample_chunk(ctx, noise, use_residual)
        a = self.clip(self.exec_slice(x0))
        out = {"latent": stats["latent"], "ratio": stats.get("ratio"), "sat": stats.get("sat"), "edit": None}
        if use_edit and self.edit is not None:
            edit, _ = self.edit(self.feat(ctx), a)
            out["edit"] = self.cfg.edit.scale * edit
            a = self.clip(a + out["edit"])
        out["a_exec"] = a
        out["q"] = self.pessimistic_q(ctx, a)
        if with_base:
            out["a_base"] = a if not use_residual and out["edit"] is None else \
                self.clip(self.exec_slice(self.sample_chunk(ctx, noise, False)[0]))
        return out

    # ---------------------------------------------------------------- losses
    def critic_loss(self, ctx: Context, action: Tensor, reward: Tensor, mask: Tensor, next_ctx: Context
                    ) -> tuple[Tensor, dict]:
        q_cfg = self.cfg.qam
        with torch.no_grad():
            next_action = self.act(next_ctx, torch.randn_like(self._noise_like(next_ctx)))["a_exec"]
            next_q = self.pessimistic_q(next_ctx, next_action, target=True)
            target = reward + q_cfg.discount * mask * next_q
        qs = self.critic(self.feat(ctx), action)
        loss = (qs - target).pow(2).mean()
        return loss, {
            "update/critic/loss": loss.detach(), "update/critic/q_mean": qs.mean().detach(),
            "update/critic/q_min": qs.min().detach(), "update/critic/q_max": qs.max().detach(),
            "update/critic/q_ens_std": qs.std(0, unbiased=False).mean().detach(),
            "update/critic/target_mean": target.mean(), "update/critic/td_abs": (qs.mean(0) - target).abs().mean().detach(),
            "_q": qs.detach().flatten(),
        }

    def _noise_like(self, ctx: Context) -> Tensor:
        d = self.dims
        return torch.empty(ctx.h_prefix.shape[0], d.horizon, d.pad_dim, device=ctx.h_prefix.device)

    def terminal_adjoint(self, ctx: Context, a_full: Tensor) -> Tensor:
        """-inv_temp * dQ/da on the executed slice; zero on unexecuted positions and padded dims."""
        critic = self.target_critic if self.cfg.qam.use_target_grad else self.critic
        a = self.exec_slice(a_full).detach().requires_grad_(True)
        with torch.enable_grad():
            q = critic(self.feat(ctx).detach(), self.clip(a)).mean(0).sum()
            (grad,) = torch.autograd.grad(q, a)
        adj = torch.zeros_like(a_full)
        adj[:, : self.dims.exec_horizon, : self.dims.action_dim] = -self.cfg.qam.inv_temp * grad
        return adj

    def adjoint(self, ctx: Context) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        """Memoryless rollout of the composed flow + lean adjoint through the base (qam.py adj_matching)."""
        steps, h = self.dims.flow_steps, 1.0 / self.dims.flow_steps
        x = torch.randn_like(self._noise_like(ctx))
        bsize = x.shape[0]
        xs, v_vlas, aes = [], [], []
        with torch.no_grad(), self.timer("adjoint_rollout"):
            for i in range(steps):
                s = i * h
                t = torch.full((bsize,), 1.0 - s, device=x.device)
                v_vla, hidden = self.base_fn(ctx.base, x, t)
                ae_pool = hidden.mean(1)
                xs.append(x)
                v_vlas.append(v_vla[..., : self.dims.action_dim])
                aes.append(ae_pool)
                if i != steps - 1:
                    v_phi, _ = self._residual(x, t, v_vla, ae_pool, ctx)
                    v_q = -(v_vla + self._pad(v_phi))
                    sigma = math.sqrt(2 * (1 - s + h) / (s + h))
                    x = x + h * (2 * v_q - x / (s + h)) + math.sqrt(h) * sigma * torch.randn_like(x)
                else:  # reference code: the last step is an ODE step with the base velocity only
                    x = x + h * (-v_vla)
        adj = self.terminal_adjoint(ctx, x)
        info = {"update/adjoint/terminal_abs_max": adj.abs().max(), "update/adjoint/terminal_abs_mean": adj.abs().mean(),
                "update/adjoint/terminal_abs_std": adj.abs().std(),
                "update/adjoint/dq_da_norm": adj.flatten(1).norm(dim=-1).mean() / self.cfg.qam.inv_temp
                if self.cfg.qam.inv_temp > 0 else adj.new_zeros(())}
        adjs = [None] * steps
        with self.timer("vjp"):
            self._lean_adjoint(ctx, xs, adj, adjs, h)
        return torch.stack(xs), torch.stack(adjs), torch.stack(v_vlas), torch.stack(aes), \
            torch.arange(steps, device=x.device) * h, info

    def _lean_adjoint(self, ctx: Context, xs: list, adj: Tensor, adjs: list, h: float) -> None:
        bsize = adj.shape[0]
        for i in reversed(range(len(xs))):
            s = i * h
            xi = xs[i].detach().requires_grad_(True)
            # Intentionally t_q = s + h (one step later than the forward drift), as in qam.py adj_matching:96.
            # Linearizing at s instead steered less on the analytic toy (mean 0.52 vs 0.58 at K=10, target 0.667).
            t = torch.full((bsize,), 1.0 - (s + h), device=adj.device)
            with torch.enable_grad():
                v, _ = self.base_fn(ctx.base, xi, t)
                drift = 2 * (-v) - xi / (s + h)  # lean adjoint: base drift only
                (vjp,) = torch.autograd.grad(drift, xi, grad_outputs=adj)
            adj = adj + h * vjp
            adjs[i] = adj

    def residual_loss(self, ctx: Context) -> tuple[Tensor, dict, dict]:
        xs, adjs, v_vlas, aes, ss, info = self.adjoint(ctx)
        steps, bsize = xs.shape[:2]
        h, a_dim = 1.0 / steps, self.dims.action_dim
        t = (1.0 - ss).repeat_interleave(bsize)
        v_phi, sat = self.residual(xs[..., :a_dim].flatten(0, 1), t, v_vlas.flatten(0, 1), aes.flatten(0, 1),
                                   ctx.h_prefix.repeat(steps, 1), ctx.state.repeat(steps, 1))
        fine_q = -v_phi.view(steps, bsize, *v_phi.shape[1:])
        sigma = torch.sqrt(2 * (1 - ss + h) / (ss + h)).view(steps, 1, 1, 1)
        per_step = (fine_q * 2 / sigma + sigma * adjs[..., :a_dim]).pow(2).sum((-1, -2))  # [K, B]; padded dims masked
        loss = per_step.sum(0).mean()
        ratio = v_phi.norm(dim=-1).mean(-1).view(steps, bsize) / (v_vlas.norm(dim=-1).mean(-1) + 1e-8)
        info.update({"update/residual/adj_loss": loss.detach(), "update/residual/cap_sat_frac": sat.float().mean(),
                     "update/residual/adj_loss_step": per_step.mean(1).detach(),
                     "update/residual/vphi_ratio_step": ratio.mean(1).detach(),
                     "update/adjoint/abs_mean_step": adjs[..., :a_dim].abs().mean((1, 2, 3))})
        hist = {"vphi_ratio": ratio.detach().flatten(), "adj_terminal_norm": adjs[-1].flatten(1).norm(dim=-1)}
        return loss, info, hist

    def edit_losses(self, ctx: Context) -> tuple[Tensor, Tensor, dict, dict]:
        with torch.no_grad():
            x0, _ = self.sample_chunk(ctx, torch.randn_like(self._noise_like(ctx)), self.cfg.qam.inv_temp > 0)
            chunk = self.clip(self.exec_slice(x0))
        feat = self.feat(ctx).detach()
        edit, log_prob = self.edit(feat, chunk)
        edited = self.clip(chunk + self.cfg.edit.scale * edit)
        with frozen(self.critic):
            q = self.critic(feat, edited).mean(0)
        alpha = self.edit.log_alpha.exp()
        entropy = -log_prob.detach().mean()
        q_loss = -q.mean()
        actor_loss = q_loss + (log_prob * alpha.detach()).mean()
        alpha_loss = alpha * (entropy - self.target_entropy)
        info = {"update/edit/q_loss": q_loss.detach(), "update/edit/entropy": entropy,
                "update/edit/target_entropy": torch.tensor(self.target_entropy), "update/edit/alpha": alpha.detach(),
                "update/edit/alpha_loss": alpha_loss.detach(),
                "update/edit/abs_norm": (self.cfg.edit.scale * edit).abs().mean().detach()}
        return actor_loss, alpha_loss, info, {"edit": (self.cfg.edit.scale * edit).detach().flatten()}

    def update(self, ctx: Context, action: Tensor, reward: Tensor, mask: Tensor, next_ctx: Context
               ) -> tuple[dict, dict]:
        """One QAM-E step: all losses use the pre-update parameters (as in QAM's joint loss), then each module steps."""
        with self.timer("critic"):
            critic_loss, info = self.critic_loss(ctx, action, reward, mask, next_ctx)
        residual_loss, res_info, hist = self.residual_loss(ctx)
        info.update(res_info)
        losses = {"critic": critic_loss, "residual": residual_loss}
        if self.edit is not None:
            with self.timer("edit"):
                losses["edit"], losses["alpha"], edit_info, edit_hist = self.edit_losses(ctx)
            info.update(edit_info)
            hist.update(edit_hist)
        for opt in self.optims.values():
            opt.zero_grad(set_to_none=True)
        with self.timer("backward"):
            sum(losses.values()).backward()
        max_norm = self.cfg.qam.clip_grad_norm
        for name, opt in self.optims.items():
            params = [p for g in opt.param_groups for p in g["params"]]
            info[f"update/opt/{name}_grad_norm"] = torch.nn.utils.clip_grad_norm_(
                params, max_norm if max_norm is not None else float("inf"))
        if not all(torch.isfinite(v).all() for v in info.values() if isinstance(v, Tensor)):
            return info, hist  # no step: the caller's NaN guard dumps the uncorrupted pre-step state
        for opt in self.optims.values():
            opt.step()
        with torch.no_grad():
            tau = self.cfg.qam.tau
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters(), strict=True):
                tp.lerp_(p, tau)
        hist["q"] = info.pop("_q", torch.zeros(0))
        return info, hist

    # ---------------------------------------------------------------- checkpoint helpers
    def optim_state_dict(self) -> dict:
        return {k: o.state_dict() for k, o in self.optims.items()}

    def load_optim_state_dict(self, state: dict) -> None:
        for k, o in self.optims.items():
            o.load_state_dict(state[k])

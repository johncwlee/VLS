"""Residual RL on a frozen flow policy (π0.5), in one of two `actor.mode`s sharing the critic and the buffer.

action_residual (default): a deterministic bounded correction delta = s * tanh(.) is added to π0.5's executed chunk
and trained to maximize the critic (residual RL with the RLPD recipe: 50:50 batches, LayerNorm critic ensemble).
Updates need no π0.5 pass: the buffer stores π0.5's chunk a_base for every observation, so the critic target uses
clip(a_base' + delta(o', a_base')) and the actor loss uses the stored a_base.

qam_velocity: QAM-E as run on 2026-11-10 (tag residual-qam-e-v1), described below. It cannot move a nearly
deterministic base: the tilt it allows scales with π0.5's per-observation variance (~0.004).

QAM-E: Q-learning with Adjoint Matching plus an edit policy, for a velocity residual on a frozen flow policy.
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
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from residual.networks import ActionResidual, CriticEnsemble, EditActor, ResidualMLP

# (base state, x [B, H, D], t [B]) -> (velocity [B, H, D], action-expert hidden [B, H, D_ae]), openpi convention
BaseFn = Callable[[Any, Tensor, Tensor], tuple[Tensor, Tensor]]
MODES = ("action_residual", "qam_velocity")
FEAT_STD_FLOOR = 1e-3  # critic/actor feature standardization; the smallest h_prefix dim std measured is 0.056


@dataclass
class Context:
    """Per-observation inputs. `base` is opaque state for the base flow (π0.5's prefix KV cache and masks)."""
    h_prefix: Tensor  # [B, D_p] pooled prefix features under the original instruction
    state: Tensor     # [B, S] normalized proprioception
    progress: Tensor  # [B] chunk index / max episode chunks
    base: Any
    a_base: Tensor | None = None  # [B, n_exec, A] π0.5's executed chunk here (stored; action_residual updates)


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
        if cfg.actor.mode not in MODES:
            raise ValueError(f"actor.mode={cfg.actor.mode!r}; expected one of {MODES}")
        exec_dim = dims.exec_horizon * dims.action_dim
        self.feat_dim = dims.prefix_dim + dims.state_dim + int(cfg.critic.time_feature)
        # feature standardization, fitted once on the offline pool (fit_feat_stats); identity until then
        self.register_buffer("feat_mean", torch.zeros(self.feat_dim))
        self.register_buffer("feat_std", torch.ones(self.feat_dim))
        # action_residual: the critic also gets the residual itself (critic_feat), exec_dim more inputs
        critic_in = self.feat_dim + exec_dim * (2 if cfg.actor.mode == "action_residual" else 1)
        self.critic = CriticEnsemble(critic_in, cfg.critic)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.residual = self.edit = self.actor = None
        groups = {"critic": self.critic.parameters()}
        if cfg.actor.mode == "action_residual":
            if cfg.actor.bound <= 0:
                raise ValueError(f"actor.bound must be > 0, got {cfg.actor.bound}")
            self.actor = ActionResidual(self.feat_dim, dims.exec_horizon, self.bound_vector(), cfg.actor)
            groups["actor"] = self.actor.parameters()
        else:
            self.residual = ResidualMLP(dims.horizon, dims.action_dim, dims.ae_dim, dims.prefix_dim, dims.state_dim,
                                        cfg.residual)
            self.edit = EditActor(self.feat_dim, exec_dim, cfg.edit) if cfg.edit.scale > 0 else None
            self.target_entropy = -cfg.edit.target_entropy_multiplier * exec_dim
            groups["residual"] = self.residual.parameters()
            if self.edit is not None:
                groups["edit"] = [p for n, p in self.edit.named_parameters() if n != "log_alpha"]
                groups["alpha"] = [self.edit.log_alpha]
        self.optims = {k: torch.optim.Adam(list(v), lr=cfg.qam.lr) for k, v in groups.items()}

    # ---------------------------------------------------------------- helpers
    def raw_feat(self, ctx: Context) -> Tensor:
        parts = [ctx.h_prefix, ctx.state] + ([ctx.progress[:, None]] if self.cfg.critic.time_feature else [])
        return torch.cat(parts, dim=-1)

    def feat(self, ctx: Context) -> Tensor:
        """Critic / actor input: pooled prefix features + state (+ progress), standardized per dim. Raw prefix
        features have norm ~34 with a constant offset of ~33 and 2 dims carrying most of the variance."""
        return (self.raw_feat(ctx) - self.feat_mean) / self.feat_std

    @torch.no_grad()
    def fit_feat_stats(self, raw: Tensor) -> None:
        self.feat_mean.copy_(raw.mean(0))
        self.feat_std.copy_(raw.std(0).clamp_min(FEAT_STD_FLOOR))

    def bound_vector(self) -> Tensor:
        """Per action dim: s (actor.bound) on the pose dims, actor.gripper_bound on the last (gripper)."""
        b = torch.full((self.dims.action_dim,), float(self.cfg.actor.bound))
        b[-1] = float(self.cfg.actor.gripper_bound)
        return b.to(self.action_low.device)

    def explore_std(self) -> Tensor:
        """Std of the collection-time exploration offset per action dim: explore.std_frac * s, 0 where s is 0."""
        return self.cfg.explore.std_frac * self.bound_vector()

    def clip(self, a: Tensor) -> Tensor:
        return torch.clamp(a, self.action_low, self.action_high) if self.cfg.qam.clip_actions else a

    def exec_slice(self, x: Tensor) -> Tensor:
        return x[:, : self.dims.exec_horizon, : self.dims.action_dim]

    def _residual(self, x: Tensor, t: Tensor, v_vla: Tensor, ae_pool: Tensor, ctx: Context) -> tuple[Tensor, Tensor]:
        a = self.dims.action_dim
        return self.residual(x[..., :a], t, v_vla[..., :a], ae_pool, ctx.h_prefix, ctx.state)

    def _pad(self, v_phi: Tensor) -> Tensor:
        return F.pad(v_phi, (0, self.dims.pad_dim - self.dims.action_dim))

    def critic_feat(self, ctx: Context, action: Tensor) -> Tensor:
        """The critic's non-action input. action_residual appends the residual (action - ctx.a_base) / s, the only
        part of the action that exploration and the actor vary (unit scale under exploration, exactly zero on π0.5's
        own chunks): inside `action` it is an s-sized offset on top of a_base, which the critic cannot see
        separately. Differentiable in `action`."""
        feat = self.feat(ctx)
        if self.actor is None:
            return feat
        bound = self.actor.bound
        residual = (action - ctx.a_base) / torch.where(bound > 0, bound, 1.0)
        return torch.cat([feat, residual.flatten(1)], dim=-1)

    def pessimistic_q(self, ctx: Context, action: Tensor, target: bool = False) -> Tensor:
        qs = (self.target_critic if target else self.critic)(self.critic_feat(ctx, action), action)
        return qs.mean(0) - self.cfg.qam.rho * qs.std(0, unbiased=False)

    # ---------------------------------------------------------------- sampling (openpi convention)
    @torch.no_grad()
    def sample_chunk(self, ctx: Context, noise: Tensor, use_residual: bool, with_base: bool = False
                     ) -> tuple[Tensor, dict]:
        """Euler ODE t: 1 -> 0 exactly as lerobot's PI05Pytorch.sample_actions, plus v_phi when use_residual.
        with_base also integrates the plain base flow from the same noise as extra batch rows of the same base calls
        (cheaper than a second sample); it is returned as stats["x_base"]. The base state must support .repeat(n)."""
        steps, bsize = self.dims.flow_steps, noise.shape[0]
        dt = torch.tensor(-1.0 / steps, dtype=torch.float32, device=noise.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=noise.device)
        base, x = ctx.base, noise
        if with_base:
            base, x = (base.repeat(2) if base is not None else None), torch.cat([noise, noise])
        latent, ratios, sats = None, [], []
        while time >= -dt / 2:
            t = time.expand(x.shape[0])
            v, hidden = self.base_fn(base, x, t)
            ae_pool = hidden[:bsize].mean(1)
            latent = ae_pool if latent is None else latent  # first flow step: RL²-VLA / SAFE latent
            if use_residual:
                v_own = v[:bsize]
                v_phi, sat = self._residual(x[:bsize], t[:bsize], v_own, ae_pool, ctx)
                ratios.append(v_phi.norm(dim=-1).mean() / (v_own[..., : self.dims.action_dim].norm(dim=-1).mean() + 1e-8))
                sats.append(sat.float().mean())
                v = torch.cat([v_own + self._pad(v_phi), v[bsize:]])
            x = x + dt * v
            time = time + dt
        stats = {"latent": latent}
        if use_residual:
            stats.update(ratio=torch.stack(ratios).mean(), sat=torch.stack(sats).mean())
        if with_base:
            stats["x_base"] = x[bsize:]
        return x[:bsize], stats

    @torch.no_grad()
    def act(self, ctx: Context, noise: Tensor, use_residual: bool = True, with_base: bool = False,
            explore_std: Tensor | None = None, generator: torch.Generator | None = None) -> dict:
        """Executed chunk [B, n_exec, A] (normalized, clipped), plus diagnostics and the same-noise base chunk.
        use_residual applies the learned policy (the action residual, or the velocity residual and QAM-E's edit);
        without it the chunk is π0.5's. explore_std [A] adds one Gaussian offset per action
        dim, held over the chunk (collection only; noise drawn per step would largely cancel within a chunk).
        `generator` seeds QAM-E's edit sample (seeded evaluation)."""
        if self.actor is not None:
            x0, stats = self.sample_chunk(ctx, noise, use_residual=False)
            a_base = self.clip(self.exec_slice(x0))
            ctx = replace(ctx, a_base=a_base)
            out = {"latent": stats["latent"], "ratio": None, "sat": None, "edit": None, "a_base": a_base}
            a = a_base
            if use_residual:
                out["edit"] = self.actor(self.feat(ctx), a_base)
                a = a + out["edit"]
            if explore_std is not None:
                a = a + torch.randn(a.shape[0], 1, a.shape[-1], device=a.device) * explore_std
            out["a_exec"] = a = self.clip(a)
            out["q"] = self.pessimistic_q(ctx, a)
            return out
        use_edit = use_residual and self.edit is not None
        use_residual = use_residual and self.cfg.qam.inv_temp > 0  # QAM: inv_temp == 0 samples the base only
        x0, stats = self.sample_chunk(ctx, noise, use_residual, with_base=with_base and use_residual)
        a = self.clip(self.exec_slice(x0))
        out = {"latent": stats["latent"], "ratio": stats.get("ratio"), "sat": stats.get("sat"), "edit": None}
        if use_edit:
            edit, _ = self.edit(self.feat(ctx), a, generator)
            out["edit"] = self.cfg.edit.scale * edit
            a = self.clip(a + out["edit"])
        out["a_exec"] = a
        out["q"] = self.pessimistic_q(ctx, a)
        if with_base:  # without the residual, x0 is already the base chunk
            out["a_base"] = self.clip(self.exec_slice(stats.get("x_base", x0)))
        return out

    # ---------------------------------------------------------------- losses
    def critic_loss(self, ctx: Context, action: Tensor, reward: Tensor, mask: Tensor, next_ctx: Context
                    ) -> tuple[Tensor, dict]:
        q_cfg = self.cfg.qam
        with torch.no_grad():
            if self.actor is not None:  # the policy's own noise-free action at s'
                next_action = self.policy_action(next_ctx)
            else:
                next_action = self.act(next_ctx, self.noise(next_ctx))["a_exec"]
            next_q = self.pessimistic_q(next_ctx, next_action, target=True)
            target = reward + q_cfg.discount * mask * next_q
        qs = self.critic(self.critic_feat(ctx, action), action)
        loss = (qs - target).pow(2).mean()
        return loss, {
            "update/critic/loss": loss.detach(), "update/critic/q_mean": qs.mean().detach(),
            "update/critic/q_min": qs.min().detach(), "update/critic/q_max": qs.max().detach(),
            "update/critic/q_ens_std": qs.std(0, unbiased=False).mean().detach(),
            "update/critic/target_mean": target.mean(), "update/critic/td_abs": (qs.mean(0) - target).abs().mean().detach(),
            "_q": qs.detach().flatten(),
        }

    def noise(self, ctx: Context, generator: torch.Generator | None = None) -> Tensor:
        """Flow start x_1 ~ N(0, I) over the padded chunk, one per batch row. A per-episode generator keeps π0.5's
        noise sequence independent of every other random draw (seeded, paired evaluation)."""
        d = self.dims
        return torch.randn(ctx.h_prefix.shape[0], d.horizon, d.pad_dim, device=ctx.h_prefix.device,
                           generator=generator)

    def policy_action(self, ctx: Context) -> Tensor:
        """action_residual: clip(a_base + delta(o, a_base)), differentiable in the actor."""
        return self.clip(ctx.a_base + self.actor(self.feat(ctx), ctx.a_base))

    def actor_loss(self, ctx: Context) -> tuple[Tensor, dict, dict]:
        """Maximize the critic's ensemble mean at the corrected chunk; the critic is frozen for this loss."""
        feat = self.feat(ctx)
        delta = self.actor(feat, ctx.a_base)
        action = self.clip(ctx.a_base + delta)
        with frozen(self.critic):
            q = self.critic(self.critic_feat(ctx, action), action).mean(0)
        loss = -q.mean()
        frac = self._bound_frac(delta)
        info = {"update/actor/q_loss": loss.detach(), "update/actor/abs_frac": frac.mean(),
                "update/actor/sat_frac": (frac > 0.95).float().mean()}
        return loss, info, {"actor_delta": delta.detach().flatten()}

    def _bound_frac(self, delta: Tensor) -> Tensor:
        """|delta| / s on the dims the residual may move."""
        bound = self.actor.bound
        return (delta.detach().abs() / bound.clamp_min(1e-8))[..., bound > 0]

    @torch.no_grad()
    def action_metrics(self, ctx: Context, a_exec: Tensor, rtg: Tensor) -> dict:
        """Does the critic tell actions apart (logged as gate/*)? On held-out chunks executed with
        exploration noise:
        gate/q_action_diff = mean |Q(o, a_exec) - Q(o, a_base)|; gate/action_skill = 1 - MSE(Q(o, a_exec), G) /
        MSE(Q(o, a_base), G), > 0 when the executed action improves the return prediction (approximate: Q is for the
        noise-free policy, the return G was realized with exploration)."""
        q_exec, q_base = self.pessimistic_q(ctx, a_exec), self.pessimistic_q(ctx, ctx.a_base)
        with torch.enable_grad():
            a = ctx.a_base.detach().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.critic(self.critic_feat(ctx, a), a).mean(0).sum(), a)
        mse_base = (q_base - rtg).pow(2).mean().clamp_min(1e-12)
        out = {"gate/q_action_diff": (q_exec - q_base).abs().mean(), "gate/dq_da_norm": grad.flatten(1).norm(dim=-1).mean(),
               "gate/action_skill": 1 - (q_exec - rtg).pow(2).mean() / mse_base}
        if self.actor is not None:
            frac = self._bound_frac(self.actor(self.feat(ctx), ctx.a_base))
            out.update({"gate/residual_frac": frac.mean(), "gate/residual_sat_frac": (frac > 0.95).float().mean()})
        return out

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
        x = self.noise(ctx)
        bsize = x.shape[0]
        xs, v_vlas, aes = [], [], []
        with torch.no_grad():
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
            x0, _ = self.sample_chunk(ctx, self.noise(ctx), self.cfg.qam.inv_temp > 0)
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

    def update(self, ctx: Context, action: Tensor, reward: Tensor, mask: Tensor, next_ctx: Context,
               actor: bool = False) -> tuple[dict, dict]:
        """One step: all losses use the pre-update parameters (as in QAM's joint loss), then each module steps.
        action_residual trains the actor only when `actor` (off during the critic warm-up; every rank must pass
        the same value); qam_velocity always trains the residual and the edit."""
        critic_loss, info = self.critic_loss(ctx, action, reward, mask, next_ctx)
        losses, hist = {"critic": critic_loss}, {}
        if self.actor is not None:
            if actor:
                losses["actor"], actor_info, hist = self.actor_loss(ctx)
                info.update(actor_info)
        else:
            losses["residual"], res_info, hist = self.residual_loss(ctx)
            info.update(res_info)
            if self.edit is not None:
                losses["edit"], losses["alpha"], edit_info, edit_hist = self.edit_losses(ctx)
                info.update(edit_info)
                hist.update(edit_hist)
        for opt in self.optims.values():
            opt.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        if dist.is_initialized() and dist.get_world_size() > 1:
            info = self._average_over_ranks(info)
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

    def _average_over_ranks(self, info: dict) -> dict:
        """Data parallel: every loss is a mean over this rank's equal share of the batch, so the averaged gradient is
        the full-batch gradient and every rank takes the same step. Metrics are averaged too (min/max become rank
        means), so the non-finite check agrees on every rank; "_" entries are per-sample and stay local."""
        world = dist.get_world_size()
        for p in self.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world  # SUM then divide: gloo (CPU) has no AVG
        out = {}
        for k, v in info.items():
            if not k.startswith("_"):
                v = v.detach().float().to(self.action_low.device)
                dist.all_reduce(v)
                v = v / world
            out[k] = v
        return out

    # ---------------------------------------------------------------- checkpoint helpers
    def optim_state_dict(self) -> dict:
        return {k: o.state_dict() for k, o in self.optims.items()}

    def load_optim_state_dict(self, state: dict) -> None:
        for k, o in self.optims.items():
            o.load_state_dict(state[k])

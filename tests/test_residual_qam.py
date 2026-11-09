"""QAM-E residual math on a toy flow with a known answer (CPU, no π0.5).

The toy base flow maps N(0, I) noise to data N(MU0, SIG0^2 I) in openpi/lerobot convention, with an analytic
velocity. With Q(a) = -||a_exec - C||^2, QAM's target pi_base(a) * exp(inv_temp * Q(a)) is Gaussian with a closed
form mean, so the trained residual must move the executed slice there and leave the rest of the chunk alone.
The 10-step memoryless discretization undershoots the target (mean ~0.59 at K=10, ~0.62 at K=20, ~0.65 at K=40
for a target of 0.667, measured 2026-11-08); the test uses K=20 and a tolerance that covers that bias.
"""
import dataclasses

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from residual.buffer import chunk_reward
from residual.networks import cap
from residual.qam_e import Context, Dims, QAMEAgent

MU0, SIG0, C, BETA = 0.0, 1.0, 1.0, 1.0
DIMS = Dims(horizon=2, pad_dim=3, action_dim=2, exec_horizon=1, ae_dim=4, prefix_dim=4, state_dim=2, flow_steps=10)


def toy_base(base, x, t):
    """E[noise - a | x_t] for x_t = t*noise + (1-t)*a, noise ~ N(0,1), a ~ N(MU0, SIG0^2), per dim."""
    t = t.view(-1, 1, 1)
    mean, var = (1 - t) * MU0, t ** 2 + (1 - t) ** 2 * SIG0 ** 2
    v = t * (x - mean) / var - MU0 - (1 - t) * SIG0 ** 2 * (x - mean) / var
    return v, torch.zeros(*x.shape[:2], DIMS.ae_dim)


class AnalyticQ(nn.Module):
    def forward(self, feat, action):
        q = -((action.flatten(1) - C) ** 2).sum(-1)
        return torch.stack([q, q])


def make_agent(**overrides) -> QAMEAgent:
    cfg = OmegaConf.load("residual/hyperparameters.yaml")
    base = {"residual": {"hidden_dims": [128, 128], "cap_c": None}, "critic": {"hidden_dims": [32], "num_qs": 2},
            "edit": {"scale": 0.0, "hidden_dims": [32]}, "qam": {"inv_temp": BETA, "clip_actions": False, "lr": 1e-3}}
    cfg = OmegaConf.merge(cfg, base, overrides)
    bound = torch.ones(DIMS.action_dim)
    return QAMEAgent(cfg, DIMS, toy_base, -10 * bound, 10 * bound)


def context(batch: int) -> Context:
    return Context(h_prefix=torch.zeros(batch, DIMS.prefix_dim), state=torch.zeros(batch, DIMS.state_dim),
                   progress=torch.zeros(batch), base=None)


def test_residual_converges_to_tilted_gaussian():
    torch.manual_seed(0)
    agent = make_agent(qam={"lr": 5e-4})
    agent.dims = dataclasses.replace(DIMS, flow_steps=20)
    agent.target_critic = AnalyticQ()
    ctx = context(512)
    for _ in range(800):
        loss, _, _ = agent.residual_loss(ctx)
        agent.optims["residual"].zero_grad()
        loss.backward()
        agent.optims["residual"].step()
    x0, _ = agent.sample_chunk(context(8192), torch.randn(8192, DIMS.horizon, DIMS.pad_dim), use_residual=True)
    tilted_mean = (MU0 / SIG0 ** 2 + 2 * BETA * C) / (1 / SIG0 ** 2 + 2 * BETA)
    executed, unexecuted, padded = x0[:, 0, :2], x0[:, 1, :2], x0[..., 2]
    assert executed.mean().item() == pytest.approx(tilted_mean, abs=0.08)
    assert executed.var().item() == pytest.approx(1 / (1 / SIG0 ** 2 + 2 * BETA), abs=0.06)
    assert unexecuted.mean().item() == pytest.approx(MU0, abs=0.08)
    assert padded.mean().item() == pytest.approx(MU0, abs=0.05)


def test_zero_init_residual_is_exactly_the_base():
    agent = make_agent(residual={"cap_c": 0.5})
    noise = torch.randn(16, DIMS.horizon, DIMS.pad_dim)
    with_residual, _ = agent.sample_chunk(context(16), noise, use_residual=True)
    base_only, _ = agent.sample_chunk(context(16), noise, use_residual=False)
    assert torch.equal(with_residual, base_only)


def test_cap_bounds_residual_per_token():
    raw, v_vla = 10 * torch.randn(64, 50, 7), torch.randn(64, 50, 7)
    capped, saturated = cap(raw, v_vla, 0.5)
    assert (capped.norm(dim=-1) <= 0.5 * v_vla.norm(dim=-1) + 1e-5).all()
    assert saturated.float().mean() > 0.5


def test_terminal_adjoint_only_on_executed_real_dims():
    agent = make_agent()
    adj = agent.terminal_adjoint(context(8), torch.randn(8, DIMS.horizon, DIMS.pad_dim))
    assert adj[:, DIMS.exec_horizon:].abs().max() == 0
    assert adj[..., DIMS.action_dim:].abs().max() == 0
    assert adj[:, :DIMS.exec_horizon, :DIMS.action_dim].abs().max() > 0


def test_full_update_with_edit_is_finite():
    torch.manual_seed(0)
    agent = make_agent(edit={"scale": 0.1})
    ctx, next_ctx = context(16), context(16)
    action = torch.randn(16, DIMS.exec_horizon, DIMS.action_dim)
    info, hist = agent.update(ctx, action, -torch.ones(16), torch.ones(16), next_ctx)
    assert all(torch.isfinite(v).all() for v in info.values())
    assert {"q", "edit", "vphi_ratio"} <= hist.keys()


def test_chunk_reward():
    steered = np.array([False, True, False, False, True])
    terminal = np.array([False, False, True, True, True])
    success = np.array([False, False, True, False, False])
    r, mask = chunk_reward(steered, terminal, success, kappa=0.5, failure_penalty=None, t_max=10, normalize=False)
    np.testing.assert_allclose(r, [-1, -1.5, -1, -21, -21.5])
    np.testing.assert_allclose(mask, [1, 1, 0, 0, 0])
    r_norm, _ = chunk_reward(steered, terminal, success, kappa=0.5, failure_penalty=30, t_max=10, normalize=True)
    np.testing.assert_allclose(r_norm, np.array([-1, -1.5, -1, -31, -31.5]) / 10)


def test_non_finite_update_does_not_step():
    agent = make_agent(edit={"scale": 0.1})
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    action = torch.randn(4, DIMS.exec_horizon, DIMS.action_dim)
    info, _ = agent.update(context(4), action, torch.full((4,), float("nan")), torch.ones(4), context(4))
    assert not torch.isfinite(info["update/critic/loss"])
    assert all(torch.equal(before[k], v) for k, v in agent.state_dict().items())


def _split_batch_update(rank: int | None = None, world: int = 1, init_file: str = "", out: str = "") -> dict:
    """One update on a fixed batch of 8 (rank r of `world` takes its contiguous share). Noise is zeroed so the
    losses are deterministic and two half-batch ranks must match one full-batch process exactly."""
    import torch.distributed as dist
    if rank is not None:  # spawned process; the full-batch run in the pytest process uses monkeypatch
        torch.randn_like = torch.zeros_like
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    agent = make_agent()
    g = torch.Generator().manual_seed(1)
    feats = [torch.randn(8, d, generator=g) for d in (DIMS.prefix_dim, DIMS.state_dim, DIMS.prefix_dim, DIMS.state_dim)]
    action = torch.randn(8, DIMS.exec_horizon, DIMS.action_dim, generator=g)
    reward, mask = -torch.rand(8, generator=g), (torch.rand(8, generator=g) > 0.3).float()
    share = slice((rank or 0) * 8 // world, ((rank or 0) + 1) * 8 // world)
    ctx = Context(h_prefix=feats[0][share], state=feats[1][share], progress=torch.zeros(8)[share], base=None)
    next_ctx = Context(h_prefix=feats[2][share], state=feats[3][share], progress=torch.zeros(8)[share], base=None)
    info, _ = agent.update(ctx, action[share], reward[share], mask[share], next_ctx)
    result = {"params": agent.state_dict(), "critic_loss": info["update/critic/loss"].item()}
    if rank is not None:
        if rank == 0:
            torch.save(result, out)
        dist.destroy_process_group()
    return result


def test_data_parallel_update_matches_full_batch(tmp_path, monkeypatch):
    import torch.multiprocessing as mp
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    full = _split_batch_update()
    mp.spawn(_split_batch_update, args=(2, str(tmp_path / "init"), str(tmp_path / "rank0.pt")), nprocs=2)
    split = torch.load(tmp_path / "rank0.pt", weights_only=True)
    assert split["critic_loss"] == pytest.approx(full["critic_loss"], rel=1e-5)
    for k, v in full["params"].items():
        torch.testing.assert_close(split["params"][k], v, rtol=1e-5, atol=1e-6, msg=k)

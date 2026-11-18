"""Residual math on a toy flow with a known answer (CPU, no π0.5): QAM-E (actor.mode=qam_velocity) and the
action-space residual (actor.mode=action_residual), plus the buffer and the paired-evaluation statistics.

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

from residual.buffer import ReplayBuffer, chunk_reward, field_specs
from residual.monitor import paired_stats
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


def make_agent(mode: str = "qam_velocity", dims: Dims = DIMS, **overrides) -> QAMEAgent:
    cfg = OmegaConf.load("residual/hyperparameters.yaml")
    base = {"residual": {"hidden_dims": [128, 128], "cap_c": None}, "critic": {"hidden_dims": [32], "num_qs": 2},
            "edit": {"scale": 0.0, "hidden_dims": [32]}, "qam": {"inv_temp": BETA, "clip_actions": False, "lr": 1e-3},
            "actor": {"mode": mode, "hidden_dims": [64, 64], "bound": S}, "explore": {"std_frac": 0.5}}
    cfg = OmegaConf.merge(cfg, base, overrides)
    bound = torch.ones(dims.action_dim)
    return QAMEAgent(cfg, dims, toy_base, -10 * bound, 10 * bound)


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


def test_act_base_rows_match_separate_samples():
    """act(with_base) integrates the base as extra batch rows of the residual sample; both must match solo samples."""
    torch.manual_seed(0)
    agent = make_agent()
    for p in agent.residual.parameters():
        torch.nn.init.normal_(p, std=0.1)
    noise = torch.randn(4, DIMS.horizon, DIMS.pad_dim)
    out = agent.act(context(4), noise, with_base=True)
    composed, _ = agent.sample_chunk(context(4), noise, use_residual=True)
    base, _ = agent.sample_chunk(context(4), noise, use_residual=False)
    torch.testing.assert_close(out["a_exec"], agent.exec_slice(composed))
    torch.testing.assert_close(out["a_base"], agent.exec_slice(base))
    assert not torch.allclose(out["a_exec"], out["a_base"])


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


def test_buffer_resume_and_readonly_view(tmp_path):
    """A reopened buffer (resume, or a rank > 0 view) sees the same episodes and pools; a smaller capacity or another
    layout is refused."""
    specs = field_specs((3, 4, 4), DIMS)
    buf = ReplayBuffer(tmp_path, 20, specs)
    view = ReplayBuffer(tmp_path, 20, specs, readonly=True)
    chunks = {"a_exec": np.ones((3, DIMS.exec_horizon, DIMS.action_dim), np.float32), "steered": np.zeros(3, bool),
              "terminal": np.array([False, False, True]), "success": np.array([False, False, True])}
    buf.add_episode(chunks, {"task": 0}, online=False, heldout=False)
    buf.add_episode(chunks, {"task": 0}, online=True, heldout=False)
    view.reload()
    for other in (ReplayBuffer(tmp_path, 20, specs), view):
        assert other.size == 6 and len(other.episodes) == 2
        np.testing.assert_array_equal(other.protected_idx, [0, 1, 2])
        np.testing.assert_array_equal(other.online_idx, [3, 4, 5])
        np.testing.assert_array_equal(other.get(np.array([5, 0]), ["terminal"])["terminal"], [True, False])
    with pytest.raises(ValueError):
        ReplayBuffer(tmp_path, 19, specs)
    with pytest.raises(ValueError):
        ReplayBuffer(tmp_path, 20, field_specs((3, 5, 5), DIMS))
    grown = ReplayBuffer(tmp_path, 30, specs)  # a full buffer can be resumed with a larger capacity
    grown.add_episode(chunks, {"task": 0}, online=True, heldout=False)
    assert grown.size == 9 and grown.data["terminal"][8] and grown.data["terminal"][2]


def test_non_finite_update_does_not_step():
    agent = make_agent(edit={"scale": 0.1})
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    action = torch.randn(4, DIMS.exec_horizon, DIMS.action_dim)
    info, _ = agent.update(context(4), action, torch.full((4,), float("nan")), torch.ones(4), context(4))
    assert not torch.isfinite(info["update/critic/loss"])
    assert all(torch.equal(before[k], v) for k, v in agent.state_dict().items())


def _zero_noise(self, ctx: Context) -> torch.Tensor:
    return torch.zeros(ctx.h_prefix.shape[0], DIMS.horizon, DIMS.pad_dim)


def _split_batch_update(rank: int | None = None, world: int = 1, init_file: str = "", out: str = "",
                        mode: str = "qam_velocity", actor: bool = False) -> dict:
    """One update on a fixed batch of 8 (rank r of `world` takes its contiguous share). Noise is zeroed so the
    losses are deterministic and two half-batch ranks must match one full-batch process exactly."""
    import torch.distributed as dist
    if rank is not None:  # spawned process; the full-batch run in the pytest process uses monkeypatch
        torch.randn_like, QAMEAgent.noise = torch.zeros_like, _zero_noise
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    torch.manual_seed(0)
    agent = make_agent(mode)
    g = torch.Generator().manual_seed(1)
    feats = [torch.randn(8, d, generator=g) for d in (DIMS.prefix_dim, DIMS.state_dim, DIMS.prefix_dim, DIMS.state_dim)]
    action, a_base, next_a_base = (torch.randn(8, DIMS.exec_horizon, DIMS.action_dim, generator=g) for _ in range(3))
    reward, mask = -torch.rand(8, generator=g), (torch.rand(8, generator=g) > 0.3).float()
    share = slice((rank or 0) * 8 // world, ((rank or 0) + 1) * 8 // world)
    ctx = Context(h_prefix=feats[0][share], state=feats[1][share], progress=torch.zeros(8)[share], base=None,
                  a_base=a_base[share])
    next_ctx = Context(h_prefix=feats[2][share], state=feats[3][share], progress=torch.zeros(8)[share], base=None,
                       a_base=next_a_base[share])
    info, _ = agent.update(ctx, action[share], reward[share], mask[share], next_ctx, actor=actor)
    result = {"params": agent.state_dict(), "critic_loss": info["update/critic/loss"].item()}
    if rank is not None:
        if rank == 0:
            torch.save(result, out)
        dist.destroy_process_group()
    return result


@pytest.mark.parametrize("mode,actor", [("qam_velocity", False), ("action_residual", False),
                                        ("action_residual", True)])
def test_data_parallel_update_matches_full_batch(tmp_path, monkeypatch, mode, actor):
    import torch.multiprocessing as mp
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    monkeypatch.setattr(QAMEAgent, "noise", _zero_noise)
    full = _split_batch_update(mode=mode, actor=actor)
    mp.spawn(_split_batch_update, args=(2, str(tmp_path / "init"), str(tmp_path / "rank0.pt"), mode, actor), nprocs=2)
    split = torch.load(tmp_path / "rank0.pt", weights_only=True)
    assert split["critic_loss"] == pytest.approx(full["critic_loss"], rel=1e-5)
    for k, v in full["params"].items():
        torch.testing.assert_close(split["params"][k], v, rtol=1e-5, atol=1e-6, msg=k)


# ---------------------------------------------------------------- action_residual
# Toy action dims are (pose, gripper): the residual and the exploration noise may move dim 0 only.
S = 0.2  # actor.bound in these tests (pinned in make_agent with explore.std_frac, so retuned defaults do not move them)


def with_base(ctx: Context, a_base: torch.Tensor) -> Context:
    return dataclasses.replace(ctx, a_base=a_base)


class LinearQ(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, feat, action):
        q = (action * self.w).sum((-1, -2))
        return torch.stack([q, q])


class StateOnlyQ(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim  # action_residual appends the residual to feat (critic_feat): ignore it too

    def forward(self, feat, action):
        q = feat[..., :self.feat_dim].sum(-1) + 0 * (feat.sum(-1) + action.flatten(1).sum(-1))  # zero gradient
        return torch.stack([q, q])


def test_action_residual_zero_init_is_exactly_the_base():
    agent = make_agent("action_residual")
    noise = torch.randn(16, DIMS.horizon, DIMS.pad_dim)
    out = agent.act(context(16), noise, with_base=True)
    base, _ = agent.sample_chunk(context(16), noise, use_residual=False)
    assert torch.equal(out["a_exec"], agent.exec_slice(base))
    assert torch.equal(out["a_base"], out["a_exec"])
    assert out["edit"].abs().max() == 0


def test_action_residual_is_bounded_and_leaves_the_gripper():
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    for p in agent.actor.parameters():
        nn.init.normal_(p, std=3.0)
    out = agent.act(context(256), torch.randn(256, DIMS.horizon, DIMS.pad_dim), explore_std=agent.explore_std())
    delta = out["edit"]
    assert (delta.abs() <= agent.actor.bound + 1e-7).all()
    assert delta[..., 0].abs().max() > 0.9 * S
    assert torch.equal(out["a_exec"][..., -1], out["a_base"][..., -1])


def test_exploration_offset_is_held_over_the_chunk():
    torch.manual_seed(0)
    dims = dataclasses.replace(DIMS, exec_horizon=2)
    agent = make_agent("action_residual", dims=dims)
    ctx, noise = context(4096), torch.randn(4096, dims.horizon, dims.pad_dim)
    plain = agent.act(ctx, noise)["a_exec"]
    offset = agent.act(ctx, noise, explore_std=agent.explore_std())["a_exec"] - plain
    again = agent.act(ctx, noise, explore_std=agent.explore_std())["a_exec"] - plain
    torch.testing.assert_close(offset[:, 0], offset[:, 1], atol=1e-6, rtol=0)  # same offset at every step
    assert offset[..., -1].abs().max() == 0  # gripper
    assert offset[:, 0, 0].std().item() == pytest.approx(0.5 * S, rel=0.05)  # explore.std_frac * s
    assert not torch.allclose(offset, again)  # fresh offset per chunk
    assert torch.equal(agent.act(ctx, noise)["a_exec"], plain)  # off unless asked


def test_actor_climbs_a_linear_critic_to_its_bound():
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    agent.critic = LinearQ(torch.tensor([1.0, -1.0]))
    ctx = with_base(context(64), torch.randn(64, 1, 2))
    for _ in range(300):
        loss, _, _ = agent.actor_loss(ctx)
        agent.optims["actor"].zero_grad()
        loss.backward()
        agent.optims["actor"].step()
    delta = agent.actor(agent.feat(ctx), ctx.a_base)
    assert delta[..., 0].min() > 0.95 * S  # sign(w) * s
    assert delta[..., 1].abs().max() == 0


def test_critic_target_is_the_noise_free_policy_action():
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    ctx, nxt = with_base(context(8), torch.randn(8, 1, 2)), with_base(context(8), torch.randn(8, 1, 2))
    args = (ctx, torch.randn(8, 1, 2), -torch.ones(8), torch.ones(8), nxt)
    first = agent.critic_loss(*args)[1]["update/critic/target_mean"]
    assert torch.equal(first, agent.critic_loss(*args)[1]["update/critic/target_mean"])
    nn.init.normal_(agent.actor.out.weight, std=1.0)
    assert not torch.equal(first, agent.critic_loss(*args)[1]["update/critic/target_mean"])


def test_warmup_update_trains_the_critic_only():
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    ctx, nxt = with_base(context(16), torch.randn(16, 1, 2)), with_base(context(16), torch.randn(16, 1, 2))
    args = (ctx, torch.randn(16, 1, 2), -torch.ones(16), torch.ones(16), nxt)
    actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    critic_before = [p.clone() for p in agent.critic.parameters()]
    info, _ = agent.update(*args)
    assert all(torch.isfinite(v).all() for v in info.values())
    assert not any(k.startswith(("update/actor", "update/residual", "update/edit")) for k in info)
    assert all(torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters(), strict=True))
    info, _ = agent.update(*args, actor=True)
    assert "update/actor/q_loss" in info
    assert not torch.equal(actor_before["out.weight"], agent.actor.out.weight)


def test_feature_standardization():
    agent = make_agent("action_residual")
    p, s = DIMS.prefix_dim, DIMS.state_dim
    raw = torch.randn(2000, agent.feat_dim) * 5 + 3
    raw[:, 0] = 7.0  # a constant feature must stay finite
    agent.fit_feat_stats(raw)
    ctx = Context(h_prefix=raw[:, :p], state=raw[:, p:p + s], progress=raw[:, -1], base=None)
    feat = agent.feat(ctx)
    assert torch.isfinite(feat).all() and feat[:, 0].abs().max() == 0
    torch.testing.assert_close(feat[:, 1:].mean(0), torch.zeros(agent.feat_dim - 1), atol=1e-4, rtol=0)
    torch.testing.assert_close(feat[:, 1:].std(0), torch.ones(agent.feat_dim - 1), atol=1e-4, rtol=0)
    other = make_agent("action_residual")
    other.load_state_dict(agent.state_dict())  # the statistics travel with the checkpoint
    assert torch.equal(other.feat(ctx), feat)


def test_checkpoint_with_another_bound_is_refused():
    trained = make_agent("action_residual").state_dict()
    with pytest.raises(ValueError, match="bound"):
        make_agent("action_residual", actor={"bound": 0.4}).load_state_dict(trained)


def test_action_gate_separates_action_blind_and_action_aware_critics():
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    ctx = with_base(context(256), torch.randn(256, 1, 2))
    a_exec = ctx.a_base + 0.3 * torch.randn(256, 1, 2)
    rtg = -((a_exec - C) ** 2).sum((-1, -2))  # the return depends on the executed action
    agent.critic = StateOnlyQ(agent.feat_dim)
    blind = agent.action_metrics(ctx, a_exec, rtg)
    assert blind["gate/q_action_diff"] == 0 and blind["gate/dq_da_norm"] == 0
    agent.critic = AnalyticQ()
    aware = agent.action_metrics(ctx, a_exec, rtg)
    assert aware["gate/q_action_diff"] > 0
    assert aware["gate/action_skill"] > 0.5
    assert aware["gate/residual_frac"] == 0


def test_warmup_then_actor_moves_toward_the_better_action():
    """One-step task, reward peaks at a = +s on the pose dim: critic-only updates on exploration data make the
    critic predict the action's effect, then actor updates move the residual toward the better action.
    An untrained critic already has a large q_action_diff (random sensitivity, 0.79 here), so that metric alone
    does not show learning; action_skill does."""
    torch.manual_seed(0)
    agent = make_agent("action_residual", critic={"hidden_dims": [64, 64]}, qam={"lr": 3e-3})
    n = 512
    ctx = with_base(context(n), torch.zeros(n, 1, 2))
    a_exec = ctx.a_base + torch.randn(n, 1, 2) * agent.explore_std()
    reward = -10 * (a_exec[:, 0, 0] - S) ** 2
    args = (ctx, a_exec, reward, torch.zeros(n), ctx)  # terminal: target = reward
    before = agent.action_metrics(ctx, a_exec, reward)
    for _ in range(500):
        agent.update(*args)
    after = agent.action_metrics(ctx, a_exec, reward)
    assert before["gate/action_skill"] < 0.5 < after["gate/action_skill"]
    for _ in range(300):
        agent.update(*args, actor=True)
    assert agent.actor(agent.feat(ctx), ctx.a_base)[..., 0].mean() > 0.5 * S


def test_paired_stats():
    baseline = {"a": True, "b": False, "c": False, "d": True, "only_baseline": True}
    policy = {"a": True, "b": True, "c": False, "d": False, "only_policy": False}
    s = paired_stats(policy, baseline)  # differences 0, +1, 0, -1
    assert s["paired_n"] == 4 and s["paired_diff"] == 0 and s["flip_frac"] == 0.5
    assert s["paired_se"] == pytest.approx(np.sqrt(2 / 3) / 2)
    assert s["success"] == 0.5 and s["baseline_success"] == 0.5
    same = paired_stats({"a": True, "b": False}, {"a": True, "b": False})
    assert same["paired_se"] == 0 and "paired_z" not in same and same["flip_frac"] == 0
    assert paired_stats(policy, {}) == {}


def test_critic_target_value():
    """target = r + discount * mask * (mean - rho * std) of the target critic at the policy's action in s'."""
    torch.manual_seed(0)
    agent = make_agent("action_residual")
    nn.init.normal_(agent.actor.out.weight, std=1.0)
    for p in agent.target_critic.parameters():  # unlike the online critic
        nn.init.normal_(p, std=0.3)
    ctx, nxt = with_base(context(8), torch.randn(8, 1, 2)), with_base(context(8), torch.randn(8, 1, 2))
    reward, mask = -torch.rand(8), torch.tensor([1.0, 0, 1, 1, 0, 1, 1, 0])
    info = agent.critic_loss(ctx, torch.randn(8, 1, 2), reward, mask, nxt)[1]
    with torch.no_grad():
        next_action = agent.policy_action(nxt)
        qs = agent.target_critic(agent.critic_feat(nxt, next_action), next_action)
        expected = reward + agent.cfg.qam.discount * mask * (qs.mean(0) - agent.cfg.qam.rho * qs.std(0, unbiased=False))
    torch.testing.assert_close(info["update/critic/target_mean"], expected.mean())


def test_actor_loss_gives_the_critic_no_gradient():
    agent = make_agent("action_residual")
    loss, _, _ = agent.actor_loss(with_base(context(8), torch.randn(8, 1, 2)))
    loss.backward()
    assert all(p.grad is None for p in agent.critic.parameters())
    assert agent.actor.out.weight.grad.abs().sum() > 0


def test_heldout_episodes_are_never_sampled_and_terminal_rows_have_no_successor(tmp_path):
    buf = ReplayBuffer(tmp_path, 20, field_specs((3, 4, 4), DIMS))
    chunks = {"a_exec": np.ones((3, DIMS.exec_horizon, DIMS.action_dim), np.float32), "steered": np.zeros(3, bool),
              "terminal": np.array([False, False, True]), "success": np.array([False, False, True])}
    for online, heldout in ((False, False), (False, True), (True, False), (True, True)):
        buf.add_episode(chunks, {"task": 0}, online=online, heldout=heldout)
    np.testing.assert_array_equal(buf.protected_idx, [0, 1, 2])
    np.testing.assert_array_equal(buf.online_idx, [6, 7, 8])
    idx = buf.sample(200, 0.5, np.random.default_rng(0))
    assert set(idx) <= {0, 1, 2, 6, 7, 8}
    np.testing.assert_array_equal(buf.next_index(np.array([0, 1, 2, 6, 8])), [1, 2, 2, 7, 8])


def test_task_specs_per_suite_ids():
    from residual.rollout import task_specs
    env = OmegaConf.create({"suites": ["a", "b", "c"], "perturbation": "task", "task_ids": [0, 1],
                            "task_ids_by_suite": {"a": None, "b": [5], "c": []}})
    assert task_specs(env) == [("a_task", 0), ("a_task", 1), ("b_task", 5)]  # an empty list excludes the suite

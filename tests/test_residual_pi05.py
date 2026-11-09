"""π0.5 wrapper for the residual (spec §12 tests 1, 2, 9 and the buffer round trip). Needs CUDA and the cached
lerobot/pi05_libero_finetuned_v044 checkpoint; the round trip also builds a LIBERO env (slow)."""

import numpy as np
import pytest
import torch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CKPT = "lerobot/pi05_libero_finetuned_v044"


@pytest.fixture(scope="module")
def pi05():
    from residual.pi05 import Pi05
    return Pi05(CKPT, "cuda")


@pytest.fixture(scope="module")
def agent(pi05):
    from residual.qam_e import QAMEAgent
    cfg = OmegaConf.load("residual/hyperparameters.yaml")
    dims = pi05.dims(cfg.policy.exec_horizon, cfg.policy.num_inference_steps)
    return QAMEAgent(cfg, dims, pi05.velocity_and_hidden, pi05.action_low, pi05.action_high).cuda()


@pytest.fixture(scope="module")
def batch(pi05):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (2, 3, 256, 256), dtype=np.uint8)
    image2 = rng.integers(0, 256, (2, 3, 256, 256), dtype=np.uint8)
    state = rng.normal(0, 0.1, (2, 8)).astype(np.float32)
    return pi05.preprocess(image, image2, state, ["put the bowl on the plate"] * 2)


def noise_for(agent, n=2):
    d = agent.dims
    return torch.randn(n, d.horizon, d.pad_dim, device="cuda", generator=torch.Generator("cuda").manual_seed(0))


def test_zero_residual_sampler_matches_stock_sample_actions(pi05, agent, batch):
    ctx = pi05.context(batch, torch.zeros(2))
    noise = noise_for(agent)
    ours, _ = agent.sample_chunk(ctx, noise, use_residual=True)  # zero-init residual
    images, img_masks = pi05.policy._preprocess_images(batch)
    stock = pi05.model.sample_actions(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                      batch[OBS_LANGUAGE_ATTENTION_MASK], noise=noise.clone())
    torch.testing.assert_close(ours, stock, atol=1e-5, rtol=0)


def test_qam_convention_base_rollout_matches_openpi_sampling(pi05, agent, batch):
    ctx = pi05.context(batch, torch.zeros(2))
    noise = noise_for(agent)
    steps, h = agent.dims.flow_steps, 1.0 / agent.dims.flow_steps
    x = noise
    with torch.no_grad():
        for i in range(steps):  # QAM: s = i*h, v_q(x, s) = -v_openpi(x, 1 - s), x += h * v_q
            v, _ = pi05.velocity_and_hidden(ctx.base, x, torch.full((2,), 1.0 - i * h, device="cuda"))
            x = x + h * (-v)
    openpi, _ = agent.sample_chunk(ctx, noise, use_residual=False)
    torch.testing.assert_close(x, openpi, atol=5e-3, rtol=0)  # time grid 1-i*h vs accumulated, through bf16


def test_vjp_through_action_expert_matches_finite_difference(pi05, agent, batch):
    """Per-direction finite differences are too noisy in bf16 (errors up to ~70%), so compare the directional
    derivatives over 16 random directions: measured 2026-11-08 corr 0.9997, slope 0.99 at eps 0.1."""
    torch.manual_seed(0)
    ctx = pi05.context(batch, torch.zeros(2))
    x = noise_for(agent).requires_grad_(True)
    t = torch.full((2,), 0.5, device="cuda")
    v, _ = pi05.velocity_and_hidden(ctx.base, x, t)
    weights = torch.randn_like(v)
    (grad,) = torch.autograd.grad((v * weights).sum(), x)
    eps, analytic, finite = 0.1, [], []
    with torch.no_grad():
        for _ in range(16):
            direction = torch.randn_like(x)
            f_plus = (pi05.velocity_and_hidden(ctx.base, x + eps * direction, t)[0] * weights).sum()
            f_minus = (pi05.velocity_and_hidden(ctx.base, x - eps * direction, t)[0] * weights).sum()
            finite.append(((f_plus - f_minus) / (2 * eps)).item())
            analytic.append((grad * direction).sum().item())
    assert np.corrcoef(analytic, finite)[0, 1] > 0.98
    assert np.polyfit(analytic, finite, 1)[0] == pytest.approx(1.0, abs=0.1)


def test_buffer_round_trip_reproduces_prefix_features(pi05):
    from residual.rollout import make_adapter, snapshot
    adapter = make_adapter({"suite_name": "libero_object"}, task_id=0)
    adapter.reset(options={"init_state": adapter._env[0]._init_states[0]})
    obs = adapter.get_policy_observation(1)
    direct = pi05.context(pi05.preprocessor(dict(obs)), torch.zeros(1))
    image, image2, state = snapshot(obs)
    rebuilt = pi05.context(pi05.preprocess(image, image2, state, obs["task"]), torch.zeros(1))
    torch.testing.assert_close(rebuilt.h_prefix, direct.h_prefix, atol=0, rtol=0)
    torch.testing.assert_close(rebuilt.state, direct.state, atol=0, rtol=0)

"""One LIBERO episode with π0.5 + residual (+ edit) through the VLS LiberoAdapter, mirroring main.py's loop:
a new chunk every `exec_horizon` steps, unnormalized by the policy postprocessor, gripper binarized by the
adapter, success/termination from the env (spec §7.1 labels: `label_source = "sim"` until the VLM judge exists).
"""

import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.constants import OBS_STATE
from omegaconf import OmegaConf

from core.env_adapters import create_adapter
from residual.pi05 import IMAGE_KEYS, Pi05
from residual.qam_e import QAMEAgent
from utils.vis_utils import add_text_to_image

VLS_ROOT = Path(__file__).resolve().parents[1]
LABEL_SOURCE = "sim"  # env check_success; the agent phase swaps in the VLM completion judge


def make_adapter(env_overrides: dict, task_id: int):
    """One adapter per task, so train/eval code picks the task and the init state explicitly."""
    cfg = OmegaConf.to_container(OmegaConf.load(VLS_ROOT / "configs/backend/libero.yaml").libero, resolve=True)
    cfg.update(env_overrides)
    cfg.update(task_ids_filter=[task_id], episode_num=1, start_episode=0)
    return create_adapter("libero", cfg)


def task_specs(env_cfg) -> list[tuple[str, int]]:
    """(suite, task id) for every task of a run. One run covers one perturbation type across base suites:
    suite = f"{base}_{env.perturbation}"; tasks are env.task_ids_by_suite[base] if given, else env.task_ids."""
    specs = []
    for base in env_cfg.suites:
        ids = env_cfg.task_ids_by_suite.get(base) or env_cfg.task_ids
        specs += [(f"{base}_{env_cfg.perturbation}", int(t)) for t in ids]
    return specs


class EnvPool:
    """Least-recently-used cache of LIBERO adapters: each holds ~0.5-1.5 GB RAM, and rebuilding one takes 1-2 s."""

    def __init__(self, max_open: int):
        self.max_open, self.open = max_open, OrderedDict()

    def get(self, suite: str, task_id: int):
        key = (suite, task_id)
        if key in self.open:
            self.open.move_to_end(key)
            return self.open[key]
        while len(self.open) >= self.max_open:
            _, old = self.open.popitem(last=False)
            for env in old._env:
                env.close()
        self.open[key] = make_adapter({"suite_name": suite}, task_id)
        return self.open[key]


def snapshot(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Env-preprocessed observation -> (uint8 camera, uint8 wrist camera, raw state); exact inverse of /255."""
    def to_u8(img):
        return (img * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
    return to_u8(obs[IMAGE_KEYS[0]]), to_u8(obs[IMAGE_KEYS[1]]), obs[OBS_STATE].float().numpy()


@dataclass
class Episode:
    chunks: dict[str, np.ndarray]  # replay-buffer fields, one row per chunk
    success: bool
    steps: int
    q: list[float]                 # pessimistic Q of each executed chunk (critic-alarm calibration later)
    stats: dict = field(default_factory=dict)
    frames: list[np.ndarray] | None = None


def run_episode(adapter, pi05: Pi05, agent: QAMEAgent, *, task_idx: int, init_state: np.ndarray, t_max: int,
                use_residual: bool, use_edit: bool, record_video: bool = False) -> Episode:
    adapter.reset(options={"init_state": init_state})
    dims, std = agent.dims, pi05.action_std.numpy()
    rows, frames, q_values = defaultdict(list), ([] if record_video else None), []
    steps, chunk, info, policy_time = 0, 0, {}, 0.0
    while True:
        obs = adapter.get_policy_observation(1)
        image, image2, state = snapshot(obs)
        start = time.perf_counter()
        batch = pi05.preprocess(image, image2, state, obs["task"])
        ctx = pi05.context(batch, torch.tensor([chunk / t_max]))
        noise = torch.randn(1, dims.horizon, dims.pad_dim, device=pi05.device)
        out = agent.act(ctx, noise, use_residual=use_residual, use_edit=use_edit, with_base=True)
        raw = pi05.unnormalize(out["a_exec"])[0]
        policy_time += time.perf_counter() - start

        q_values.append(out["q"].item())
        for key, value in (("image", image[0]), ("image2", image2[0]), ("state", state[0]),
                           ("state_norm", batch[OBS_STATE][0].float().cpu().numpy()),
                           ("a_exec", out["a_exec"][0].cpu().numpy()), ("a_base", out["a_base"][0].cpu().numpy()),
                           ("noise", noise[0].cpu().numpy().astype(np.float16)),
                           ("h_prefix", ctx.h_prefix[0].cpu().numpy().astype(np.float16)),
                           ("latent", out["latent"][0].cpu().numpy().astype(np.float16))):
            rows[key].append(value)
        for key in ("ratio", "sat"):
            if out[key] is not None:
                rows[f"_{key}"].append(out[key].item())
        if out["edit"] is not None:
            rows["_edit"].append(out["edit"].abs().mean().item())

        for k in range(dims.exec_horizon):
            if record_video:
                hud = [f"chunk {chunk} step {steps}", f"Q {q_values[-1]:.3f}",
                       f"|v_phi|/|v_vla| {rows['_ratio'][-1]:.3f}" if "_ratio" in rows else "residual off",
                       f"|edit| {rows['_edit'][-1]:.3f}" if "_edit" in rows else "edit off"]
                frames.append(add_text_to_image(np.array(adapter.get_vlm_image()), hud))
            _, _, terminated, truncated, info = adapter.step(raw[k])
            steps += 1
            if terminated or truncated:
                break
        terminal = bool(terminated or truncated)
        rows["terminal"].append(terminal)
        rows["success"].append(terminal and bool(info.get("success", False)))
        chunk += 1
        if terminal:
            break

    n = len(rows["a_exec"])
    chunks = {k: np.stack(v) for k, v in rows.items() if not k.startswith("_")}
    chunks.update(task=np.full(n, task_idx, dtype=np.int16), step=np.arange(n, dtype=np.int16),
                  steered=np.zeros(n, dtype=bool), trigger=np.zeros(n, dtype=np.uint8))
    diff = np.abs(chunks["a_exec"] - chunks["a_base"]) * std  # raw units
    stats = {"exec_vs_base_trans": diff[..., :3].mean(), "exec_vs_base_rot": diff[..., 3:6].mean(),
             "exec_vs_base_grip": diff[..., 6:].mean(), "policy_time_per_chunk": policy_time / n}
    stats.update({k[1:]: float(np.mean(v)) for k, v in rows.items() if k.startswith("_")})
    return Episode(chunks=chunks, success=bool(chunks["success"][-1]), steps=steps, q=q_values,
                   stats={k: float(v) for k, v in stats.items()}, frames=frames)

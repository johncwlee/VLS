"""Training monitor: metrics.jsonl / episodes.jsonl in the run dir always, wandb when enabled.

Every scalar row carries its step axes (update_step, env_step, round, stage) so the jsonl alone is enough for
`python -m residual.plot_run <run_dir>`, and wandb plots each metric group against the right axis.
"""

import contextlib
import csv
import json
import logging
import netrc
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

# metric prefix -> step axis
AXES = {"update": "update_step", "buffer": "update_step", "hist": "update_step", "episode": "env_step",
        "round": "round", "eval": "round", "gate": "round", "video": "round"}


# Hosts `wandb login` writes to ~/.netrc: the classic API, and W&B's CoreWeave API (wandb >= 0.30).
WANDB_NETRC_HOSTS = ("api.wandb.ai", "api.forge.coreweave.com")


def _wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    hosts = list(WANDB_NETRC_HOSTS)
    if os.environ.get("WANDB_BASE_URL"):
        hosts.append(urlparse(os.environ["WANDB_BASE_URL"]).hostname)
    try:
        entries = netrc.netrc()
    except (FileNotFoundError, netrc.NetrcParseError):
        return False
    return any(entries.authenticators(h) is not None for h in hosts)


def _scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_video(run_dir: Path, name: str, frames: list[np.ndarray], fps: int) -> Path:
    path = run_dir / "videos" / f"{name}.mp4"
    path.parent.mkdir(exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


class Monitor:
    def __init__(self, cfg, run_dir: Path):
        self.run_dir = run_dir
        self.metrics = (run_dir / "metrics.jsonl").open("a")
        self.episodes = (run_dir / "episodes.jsonl").open("a")
        self.wandb = None
        wcfg = cfg.logging.wandb
        if wcfg.mode == "disabled":
            return
        if wcfg.mode == "online" and not _wandb_credentials():
            raise SystemExit("wandb mode=online but no credentials: run `envs/vls/bin/wandb login` once (or put "
                             "WANDB_API_KEY in .env), or pass logging.wandb.mode=offline to log locally.")
        import wandb
        id_file = run_dir / "wandb_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
        id_file.write_text(run_id)
        self.wandb = wandb.init(project=wcfg.project, entity=wcfg.entity, group=wcfg.group, tags=list(wcfg.tags),
                                mode=wcfg.mode, dir=str(run_dir), id=run_id, resume="allow", name=run_dir.name,
                                config=OmegaConf.to_container(cfg, resolve=True))
        for prefix, axis in AXES.items():
            wandb.define_metric(axis)
            wandb.define_metric(f"{prefix}/*", step_metric=axis)

    def log(self, data: dict, **axes) -> None:
        row = {**axes, **{k: _scalar(v) for k, v in data.items()}}
        self.metrics.write(json.dumps(row) + "\n")
        self.metrics.flush()
        if self.wandb is not None:
            self.wandb.log(row)

    def episode(self, record: dict) -> None:
        self.episodes.write(json.dumps({k: _scalar(v) for k, v in record.items()}) + "\n")
        self.episodes.flush()

    def histogram(self, name: str, values, **axes) -> None:
        values = np.asarray(_scalar(values), dtype=np.float64).ravel()
        if values.size == 0:
            return
        p5, p50, p95 = np.percentile(values, [5, 50, 95])
        summary = {f"hist/{name}/{k}": float(v) for k, v in
                   (("mean", values.mean()), ("std", values.std()), ("p5", p5), ("p50", p50), ("p95", p95))}
        self.log(summary, **axes)
        if self.wandb is not None:
            import wandb
            self.wandb.log({f"hist/{name}": wandb.Histogram(values), **axes})

    def table(self, name: str, columns: list[str], rows: list[list], **axes) -> None:
        path = self.run_dir / "tables" / f"{name}_round{axes.get('round', 0)}.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerows([columns, *rows])
        if self.wandb is not None:
            import wandb
            self.wandb.log({f"eval/{name}": wandb.Table(columns=columns, data=rows), **axes})

    def video(self, path: Path, upload: bool, **axes) -> None:
        """Upload a video that `save_video` wrote (on any rank)."""
        if upload and self.wandb is not None:
            import wandb
            self.wandb.log({f"video/{path.stem.split('_')[0]}": wandb.Video(str(path), format="mp4"), **axes})

    def close(self) -> None:
        self.metrics.close()
        self.episodes.close()
        if self.wandb is not None:
            self.wandb.finish()


class Timer:
    """`with timer("name"):` accumulates wall time (CUDA-synchronized); `pop()` returns and clears totals."""

    def __init__(self):
        self.totals = defaultdict(float)

    @contextlib.contextmanager
    def __call__(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.totals[name] += time.perf_counter() - start

    def pop(self) -> dict:
        totals, self.totals = dict(self.totals), defaultdict(float)
        return totals

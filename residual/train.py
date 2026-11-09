"""Train the π0.5 velocity residual with QAM-E on LIBERO-PRO: spec Stages 1-3, no agent yet.

Stage 1: π0.5-only rollouts on seeded train inits -> protected pool (some episodes held out for the gate).
Stage 2: offline QAM-E updates on the protected pool; gate = critic Q vs realized return-to-go on held-out episodes.
Stage 3: rounds of [collect with π0.5 + residual + edit -> online pool; 50:50 updates; eval on official inits].
Run from the repo root: ./run_residual.sh [overrides]; resume with ./run_residual.sh train.resume_dir=<run dir>.
"""

import json
import logging
import math
import os
import platform
import random
import socket
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from residual.buffer import ReplayBuffer, chunk_reward, field_specs
from residual.monitor import Monitor, Timer
from residual.pi05 import Pi05
from residual.qam_e import Context, QAMEAgent
from residual.rollout import LABEL_SOURCE, VLS_ROOT, make_adapter, run_episode

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ENV_ERRORS = 5
VIDEO_FPS = 20


@dataclass
class Task:
    index: int        # position in env.task_ids_filter (stored in the buffer)
    task_id: int      # LIBERO task id in the suite
    name: str
    text: str
    adapter: object
    official_inits: np.ndarray
    train_inits: np.ndarray


def git_state(path: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class Trainer:
    def __init__(self, cfg, run_dir: Path, monitor: Monitor):
        self.cfg, self.run_dir, self.monitor = cfg, run_dir, monitor
        self.pi05 = Pi05(cfg.policy.pretrained_path, cfg.device)
        self.dims = self.pi05.dims(cfg.policy.exec_horizon, cfg.policy.num_inference_steps)
        self.tasks = self._make_tasks()
        max_steps = self.tasks[0].adapter._env[0]._max_episode_steps
        self.t_max = math.ceil(max_steps / cfg.policy.exec_horizon)
        self.agent = QAMEAgent(cfg, self.dims, self.pi05.velocity_and_hidden, self.pi05.action_low,
                               self.pi05.action_high).to(cfg.device)
        self.timer = Timer()
        self.agent.timer = self.timer
        if cfg.policy.residual_ckpt:
            self.agent.load_state_dict(torch.load(cfg.policy.residual_ckpt, weights_only=True)["agent"])
        libero_env = self.tasks[0].adapter._env[0]
        d = self.dims
        specs = field_specs((3, libero_env.observation_height, libero_env.observation_width), d.state_dim,
                            d.exec_horizon, d.action_dim, d.horizon, d.pad_dim, d.prefix_dim, d.ae_dim)
        self.buffer = ReplayBuffer(run_dir / "buffer", cfg.train.buffer_capacity, specs)
        self.state = {"stage": 1, "round": 0, "update_step": 0, "baseline_done": False, "best_eval": -1.0}
        self.rng = np.random.default_rng(cfg.seed)
        if (run_dir / "last.pt").exists():
            self._load(run_dir / "last.pt", full=True)
            log.info(f"Resumed {run_dir} at {self.state} with {self.buffer.size} chunks")

    def _make_tasks(self) -> list[Task]:
        cfg = self.cfg
        env_overrides = OmegaConf.to_container(cfg.env, resolve=True)
        task_ids = env_overrides.pop("task_ids_filter")
        init_dir = Path(cfg.split.train_init_dir).resolve() / cfg.env.suite_name
        tasks = []
        for index, task_id in enumerate(task_ids):
            adapter = make_adapter(env_overrides, task_id)
            libero_env = adapter._env[0]
            path = init_dir / f"{libero_env.task}.npy"
            if not path.exists():
                raise SystemExit(f"Missing train init states {path}. Run from modules/VLS: ../../envs/vls/bin/python "
                                 f"-m residual.make_train_inits env.suite_name={cfg.env.suite_name} "
                                 f"env.task_ids_filter=[{task_id}]")
            tasks.append(Task(index, task_id, libero_env.task, libero_env.task_description, adapter,
                              np.asarray(libero_env._init_states), np.load(path)))
        return tasks

    # ---------------------------------------------------------------- checkpoints
    def _save(self, name: str) -> None:
        payload = {"agent": self.agent.state_dict(), "optim": self.agent.optim_state_dict(), "state": self.state,
                   "rng": {"numpy": self.rng.bit_generator.state, "python": random.getstate(),
                           "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
        tmp = self.run_dir / f"{name}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, self.run_dir / name)

    def _load(self, path: Path, full: bool) -> None:
        payload = torch.load(path, weights_only=True)
        self.agent.load_state_dict(payload["agent"])
        self.agent.load_optim_state_dict(payload["optim"])
        if full:
            self.state = payload["state"]
            self.rng.bit_generator.state = payload["rng"]["numpy"]
            random.setstate(payload["rng"]["python"])
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])

    # ---------------------------------------------------------------- counters derived from the buffer
    @property
    def env_step(self) -> int:
        return sum(ep["steps"] for ep in self.buffer.episodes)

    def axes(self, **extra) -> dict:
        return {"update_step": self.state["update_step"], "env_step": self.env_step, "round": self.state["round"],
                "stage": self.state["stage"], **extra}

    # ---------------------------------------------------------------- rollouts
    def returns_to_go(self, chunks: dict) -> np.ndarray:
        r, _ = chunk_reward(chunks["steered"], chunks["terminal"], chunks["success"], self.cfg.reward.steer_kappa,
                            self.cfg.reward.failure_penalty, self.t_max, self.cfg.reward.normalize)
        out, g = np.zeros_like(r), 0.0
        for i in reversed(range(len(r))):  # every episode ends terminal (success, failure or timeout)
            g = r[i] + self.cfg.qam.discount * g
            out[i] = g
        return out

    def collect(self, n: int, *, stage: int, rnd: int, online: bool, use_residual: bool, use_edit: bool) -> None:
        cfg, errors, done = self.cfg, 0, 0
        while done < n:
            # Per-episode RNG keyed on the buffer's episode count: a resumed run continues the sequence instead of
            # replaying the init ids, held-out draws and noise of episodes collected after the last checkpoint.
            episode_rng = np.random.default_rng([cfg.seed, len(self.buffer.episodes), errors])
            torch.manual_seed(int(episode_rng.integers(2 ** 62)))
            task = self.tasks[len(self.buffer.episodes) % len(self.tasks)]  # round robin
            init_id = int(episode_rng.integers(len(task.train_inits)))
            video = len(self.buffer.episodes) % cfg.logging.video_train_every_episodes == 0
            start = time.perf_counter()
            try:
                ep = run_episode(task.adapter, self.pi05, self.agent, task_idx=task.index,
                                 init_state=task.train_inits[init_id], t_max=self.t_max,
                                 use_residual=use_residual, use_edit=use_edit, record_video=video)
            except Exception:  # an env failure drops this episode, not the run
                errors += 1
                self.monitor.episode({"stage": stage, "round": rnd, "task": task.task_id, "init_id": init_id,
                                      "error": traceback.format_exc()})
                log.exception(f"Episode failed (task {task.task_id}, train init {init_id}); dropped")
                if errors >= MAX_CONSECUTIVE_ENV_ERRORS:
                    raise
                continue
            errors = 0
            heldout = (not online) and episode_rng.random() < cfg.offline.heldout_frac
            ret = float(self.returns_to_go(ep.chunks)[0])
            meta = {"stage": stage, "round": rnd, "task": task.task_id, "init_id": init_id, "success": ep.success,
                    "steps": ep.steps, "label_source": LABEL_SOURCE}
            self.buffer.add_episode(ep.chunks, meta, online=online, heldout=heldout)
            done += 1
            wall = time.perf_counter() - start
            self.monitor.episode({**meta, "chunks": len(ep.q), "return": ret, "q": ep.q, "heldout": heldout,
                                  "wall_time": wall, **ep.stats})
            self.monitor.log({"episode/success": float(ep.success), "episode/length": ep.steps, "episode/return": ret,
                              "episode/task": task.task_id, "episode/init_id": init_id, "episode/wall_time": wall,
                              **{f"episode/{k}": v for k, v in ep.stats.items()}}, **self.axes())
            if ep.frames:
                self.monitor.video(f"train_ep{len(self.buffer.episodes)}_t{task.task_id}_"
                                   f"{'success' if ep.success else 'fail'}", ep.frames, VIDEO_FPS,
                                   cfg.logging.wandb.log_videos, **self.axes())

    def evaluate(self, *, use_residual: bool, use_edit: bool, tag: str) -> float:
        """Official LIBERO-PRO init states; these episodes never enter the buffer."""
        cfg = self.cfg
        rows, successes, qs, rtgs, stats = [], defaultdict(list), [], [], defaultdict(list)
        for task in self.tasks:
            for i in range(cfg.eval.episodes_per_task):
                init_id = i % len(task.official_inits)
                try:
                    ep = run_episode(task.adapter, self.pi05, self.agent, task_idx=task.index,
                                     init_state=task.official_inits[init_id], t_max=self.t_max,
                                     use_residual=use_residual, use_edit=use_edit,
                                     record_video=i < cfg.logging.video_eval_per_task)
                except Exception:
                    log.exception(f"Eval episode failed (task {task.task_id}, official init {init_id}); excluded")
                    rows.append([task.task_id, init_id, "error", 0])
                    continue
                successes[task.task_id].append(ep.success)
                rows.append([task.task_id, init_id, ep.success, ep.steps])
                qs += ep.q
                rtgs += self.returns_to_go(ep.chunks).tolist()
                for k, v in ep.stats.items():
                    stats[k].append(v)
                stats["length"].append(ep.steps)
                if ep.frames:
                    self.monitor.video(f"eval_{tag}_r{self.state['round']}_t{task.task_id}_i{init_id}_"
                                       f"{'success' if ep.success else 'fail'}", ep.frames, VIDEO_FPS,
                                       cfg.logging.wandb.log_videos, **self.axes())
        overall = float(np.mean([s for v in successes.values() for s in v])) if successes else 0.0
        metrics = {f"eval/{tag}/success": overall, **{f"eval/{tag}/success_task{t}": float(np.mean(v))
                                                       for t, v in successes.items()}}
        metrics.update({f"eval/{tag}/{k}": float(np.mean(v)) for k, v in stats.items()})
        metrics.update(self._q_vs_return(np.array(qs), np.array(rtgs), f"eval/{tag}"))
        self.monitor.log(metrics, **self.axes())
        self.monitor.table(f"{tag}_by_init", ["task", "init_id", "success", "steps"], rows, **self.axes())
        log.info(f"Eval [{tag}] round {self.state['round']}: success {overall:.3f} {dict(successes)}")
        return overall

    @staticmethod
    def _q_vs_return(q: np.ndarray, rtg: np.ndarray, prefix: str) -> dict:
        if len(q) < 2 or q.std() == 0 or rtg.std() == 0:
            return {}
        return {f"{prefix}/q_return_corr": float(np.corrcoef(q, rtg)[0, 1]), f"{prefix}/q_bias": float((q - rtg).mean())}

    @torch.no_grad()
    def gate(self) -> None:
        """Spec §10 Stage 2 gate: critic Q of held-out offline chunks vs their realized return-to-go."""
        episodes = [ep for ep in self.buffer.episodes if ep["heldout"]]
        if not episodes:
            return
        qs, rtgs, q_success, q_fail = [], [], [], []
        for ep in episodes:
            rows = np.arange(ep["start"], ep["start"] + ep["length"])
            data = self.buffer.get(rows, ["h_prefix", "state_norm", "step", "a_exec", "steered", "terminal", "success"])
            ctx = Context(h_prefix=torch.from_numpy(data["h_prefix"]).float().to(self.cfg.device),
                          state=torch.from_numpy(data["state_norm"]).to(self.cfg.device),
                          progress=torch.from_numpy(data["step"] / self.t_max).float().to(self.cfg.device), base=None)
            q = self.agent.pessimistic_q(ctx, torch.from_numpy(data["a_exec"]).to(self.cfg.device)).cpu().numpy()
            qs += q.tolist()
            rtgs += self.returns_to_go(data).tolist()
            (q_success if ep["success"] else q_fail).append(q[0])
        metrics = self._q_vs_return(np.array(qs), np.array(rtgs), "gate")
        metrics.update({"gate/q0_success_episodes": float(np.mean(q_success)) if q_success else float("nan"),
                        "gate/q0_failed_episodes": float(np.mean(q_fail)) if q_fail else float("nan")})
        self.monitor.log(metrics, **self.axes())

    # ---------------------------------------------------------------- updates
    def _context(self, rows: dict) -> Context:
        tasks = [self.tasks[i].text for i in rows["task"]]
        batch = self.pi05.preprocess(rows["image"], rows["image2"], rows["state"], tasks)
        return self.pi05.context(batch, torch.from_numpy(rows["step"] / self.t_max).float())

    def train(self, n_updates: int, offline_ratio: float, stage_ckpt: bool) -> None:
        cfg, device = self.cfg, self.cfg.device
        obs_keys = ["image", "image2", "state", "task", "step"]
        window = defaultdict(list)
        for _ in range(n_updates):
            idx = self.buffer.sample(cfg.train.batch_size, offline_ratio, self.rng)
            next_idx = self.buffer.next_index(idx)
            with self.timer("sample"):
                rows = self.buffer.get(idx, obs_keys + ["a_exec", "steered", "terminal", "success"])
                next_rows = self.buffer.get(next_idx, obs_keys)
            with self.timer("prefix_encode"):
                ctx, next_ctx = self._context(rows), self._context(next_rows)
            reward, mask = chunk_reward(rows["steered"], rows["terminal"], rows["success"], cfg.reward.steer_kappa,
                                        cfg.reward.failure_penalty, self.t_max, cfg.reward.normalize)
            with self.timer("total_agent"):
                info, hist = self.agent.update(ctx, torch.from_numpy(rows["a_exec"]).to(device),
                                               torch.from_numpy(reward).to(device), torch.from_numpy(mask).to(device),
                                               next_ctx)
            self.state["update_step"] += 1
            step = self.state["update_step"]
            if cfg.logging.nan_guard:
                self._nan_guard(info, idx, next_idx)
            for k, v in info.items():
                v = v.detach().float().cpu()
                if v.numel() == 1:
                    window[k].append(v.item())
                else:  # per-flow-step vectors
                    for i, x in enumerate(v.tolist()):
                        window[f"{k}/{i}"].append(x)
            window["buffer/batch_protected_frac"].append(float(np.isin(idx, self.buffer.protected_idx).mean()))
            window["buffer/batch_reward_mean"].append(float(reward.mean()))
            window["buffer/batch_terminal_frac"].append(float(rows["terminal"].mean()))
            if step % cfg.logging.log_every_updates == 0:
                metrics = {k: float(np.mean(v)) for k, v in window.items()}
                metrics.update({f"update/time/{k}": t / cfg.logging.log_every_updates
                                for k, t in self.timer.pop().items()})
                metrics.update({"update/time/gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                                "buffer/protected_chunks": len(self.buffer.protected_idx),
                                "buffer/online_chunks": len(self.buffer.online_idx), "update/lr": cfg.qam.lr})
                torch.cuda.reset_peak_memory_stats()
                self.monitor.log(metrics, **self.axes())
                window.clear()
            if step % cfg.logging.hist_every_updates == 0:
                for name, values in hist.items():
                    self.monitor.histogram(name, values, **self.axes())
            if stage_ckpt and step % cfg.train.ckpt_every_updates == 0:
                self._save("last.pt")

    def _nan_guard(self, info: dict, idx: np.ndarray, next_idx: np.ndarray) -> None:
        bad = [k for k, v in info.items() if not torch.isfinite(v).all()]
        if not bad:
            return
        path = self.run_dir / "debug" / f"nan_{self.state['update_step']}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"idx": idx, "next_idx": next_idx, "agent": self.agent.state_dict(),
                    "info": {k: v.detach().cpu() for k, v in info.items()}}, path)
        raise RuntimeError(f"Non-finite update metrics {bad}; batch and agent state saved to {path}")

    # ---------------------------------------------------------------- stages
    def run(self) -> None:
        cfg, s = self.cfg, self.state
        if s["stage"] == 1:
            if cfg.eval.baseline_at_start and not s["baseline_done"]:
                self.evaluate(use_residual=False, use_edit=False, tag="baseline")
                s["baseline_done"] = True
                self._save("last.pt")
            remaining = cfg.offline.episodes_per_task * len(self.tasks) - self.buffer.episode_count(stage=1)
            self.collect(remaining, stage=1, rnd=0, online=False, use_residual=False, use_edit=False)
            s["stage"] = 2
            self._save("last.pt")
        if s["stage"] == 2:
            self.train(cfg.offline.updates - s["update_step"], offline_ratio=1.0, stage_ckpt=True)
            self.gate()
            self._save("stage2.pt")
            s["stage"] = 3
            self._save("last.pt")
        while s["round"] < cfg.online.rounds:
            rnd = s["round"] + 1
            start = time.perf_counter()
            if cfg.online.reinit_each_round:
                self._load(self.run_dir / "stage2.pt", full=False)
            have = self.buffer.episode_count(stage=3, round=rnd)
            self.collect(cfg.online.episodes_per_round - have, stage=3, rnd=rnd, online=True,
                         use_residual=True, use_edit=True)
            self.train(cfg.online.updates_per_round, cfg.online.offline_ratio, stage_ckpt=False)
            s["round"] = rnd
            self._log_round(rnd, time.perf_counter() - start)
            self.gate()
            if rnd % cfg.eval.every_rounds == 0 or rnd == cfg.online.rounds:
                success = self.evaluate(use_residual=True, use_edit=True, tag="policy")
                if success > s["best_eval"]:
                    s["best_eval"] = success
                    self._save("best.pt")
            self._save("last.pt")

    def _log_round(self, rnd: int, wall: float) -> None:
        episodes = [ep for ep in self.buffer.episodes if ep["stage"] == 3 and ep["round"] == rnd]
        metrics = {"round/wall_time": wall, "round/mean_length": float(np.mean([ep["steps"] for ep in episodes]))}
        zero = 0
        for task in self.tasks:
            outcomes = [ep["success"] for ep in episodes if ep["task"] == task.task_id]
            rate = float(np.mean(outcomes)) if outcomes else float("nan")
            metrics[f"round/train_success_task{task.task_id}"] = rate
            if outcomes and rate == 0:
                zero += 1
                log.warning(f"Round {rnd}: task {task.task_id} had 0 successes; every return is identical, so QAM "
                            f"has nothing to learn from it")
        metrics["round/zero_success_tasks"] = zero
        self.monitor.log(metrics, **self.axes())


@hydra.main(config_path=".", config_name="hyperparameters", version_base="1.3")
def main(cfg) -> None:
    run_dir = Path(cfg.train.resume_dir).resolve() if cfg.train.resume_dir else Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.train.resume_dir:  # keep one train.log per run dir across restarts
        logging.getLogger().addHandler(logging.FileHandler(run_dir / "train.log"))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    meta_path = run_dir / "meta.json"
    starts = json.loads(meta_path.read_text()) if meta_path.exists() else []
    starts.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   "python": platform.python_version(), "vls": git_state(VLS_ROOT),
                   "root": git_state(VLS_ROOT.parents[1]), "overrides": list(HydraConfig.get().overrides.task)})
    meta_path.write_text(json.dumps(starts, indent=2))
    monitor = Monitor(cfg, run_dir)
    try:
        Trainer(cfg, run_dir, monitor).run()
    finally:
        monitor.close()


if __name__ == "__main__":
    main()

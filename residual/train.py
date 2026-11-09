"""Train the π0.5 velocity residual with QAM-E on LIBERO-PRO: spec Stages 1-3, no agent yet.

Stage 1: π0.5-only rollouts on seeded train inits -> protected pool (some episodes held out for the gate).
Stage 2: offline QAM-E updates on the protected pool; gate = critic Q vs realized return-to-go on held-out episodes.
Stage 3: rounds of [collect with π0.5 + residual + edit -> online pool; 50:50 updates; eval on official inits].
Run from the repo root: ./run_residual.sh [overrides]; resume with ./run_residual.sh train.resume_dir=<run dir>.
Multi-GPU (./run_residual.sh -g N, torchrun): one rank per GPU, each with its own π0.5 and envs. Collection runs in
waves of one episode per rank, eval splits the episodes over ranks, and updates split the batch over ranks with
averaged gradients. Rank 0 owns the buffer writes, logging and checkpoints.
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
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from lerobot.utils.random_utils import set_seed

from residual.buffer import ReplayBuffer, chunk_reward, field_specs
from residual.monitor import Monitor, Timer, save_video
from residual.pi05 import Pi05
from residual.qam_e import Context, QAMEAgent
from residual.rollout import LABEL_SOURCE, VLS_ROOT, EnvPool, run_episode, task_specs

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ENV_ERRORS = 5
VIDEO_FPS = 20
# Ranks wait for each other's longest episode / eval share, and for rank 0's gate and checkpoint I/O.
DIST_TIMEOUT = timedelta(hours=2)


@dataclass(eq=False)  # identity hash: tasks key per-task dicts
class Task:
    index: int        # position in the run's task list (stored in the buffer)
    suite: str        # LIBERO-PRO suite, e.g. libero_goal_task
    task_id: int      # LIBERO task id in the suite
    name: str
    text: str
    t_max: int        # max episode chunks: reward normalization and the critic's time feature
    official_inits: np.ndarray
    train_inits: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.suite}_t{self.task_id}"


def git_state(path: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


class Trainer:
    def __init__(self, cfg, run_dir: Path, monitor: Monitor | None):
        self.cfg, self.run_dir, self.monitor = cfg, run_dir, monitor  # monitor: rank 0 only
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        if cfg.train.batch_size % self.world:
            raise SystemExit(f"train.batch_size={cfg.train.batch_size} must split evenly over {self.world} GPUs")
        self.cpu_group = dist.new_group(backend="gloo", timeout=DIST_TIMEOUT) if self.world > 1 else None
        self.pi05 = Pi05(cfg.policy.pretrained_path, cfg.device)
        self.dims = self.pi05.dims(cfg.policy.exec_horizon, cfg.policy.num_inference_steps)
        self.envs = EnvPool(cfg.env.max_open_envs)
        self.tasks = self._make_tasks()
        self.t_max_by_task = np.array([t.t_max for t in self.tasks])
        self.agent = QAMEAgent(cfg, self.dims, self.pi05.velocity_and_hidden, self.pi05.action_low,
                               self.pi05.action_high).to(cfg.device)
        self.timer = Timer()
        self.agent.timer = self.timer
        if cfg.policy.residual_ckpt:
            self.agent.load_state_dict(torch.load(cfg.policy.residual_ckpt, weights_only=True)["agent"])
        libero_env = self.adapter(self.tasks[0])._env[0]
        specs = field_specs((3, libero_env.observation_height, libero_env.observation_width), self.dims)
        if self.rank == 0:
            self.buffer = ReplayBuffer(run_dir / "buffer", cfg.train.buffer_capacity, specs)
        self._barrier()
        if self.rank > 0:
            self.buffer = ReplayBuffer(run_dir / "buffer", cfg.train.buffer_capacity, specs, readonly=True)
        self.state = {"stage": 1, "round": 0, "update_step": 0, "baseline_done": False, "best_eval": -1.0}
        self.rng = np.random.default_rng(cfg.seed)
        if (run_dir / "last.pt").exists():
            self._load(run_dir / "last.pt", full=True)
            log.info(f"Resumed {run_dir} at {self.state} with {self.buffer.size} chunks")
        if self.world > 1:
            for t in self.agent.state_dict().values():  # identical replicas (also params and buffers)
                dist.broadcast(t, src=0)
            if self.rank > 0:  # independent update noise per rank (a resume restored rank 0's RNG everywhere)
                torch.manual_seed(int(np.random.default_rng([cfg.seed, self.rank, self.state["update_step"]])
                                      .integers(2 ** 62)))

    def _make_tasks(self) -> list[Task]:
        cfg = self.cfg
        init_dir = Path(cfg.split.train_init_dir).resolve()
        tasks = []
        for index, (suite, task_id) in enumerate(task_specs(cfg.env)):
            libero_env = self.envs.get(suite, task_id)._env[0]
            path = init_dir / suite / f"{libero_env.task}.npy"
            if not path.exists():
                raise SystemExit(f"Missing train init states {path}. Run from modules/VLS with the same env.* "
                                 f"overrides: ../../envs/vls/bin/python -m residual.make_train_inits")
            t_max = math.ceil(libero_env._max_episode_steps / cfg.policy.exec_horizon)
            tasks.append(Task(index, suite, task_id, libero_env.task, libero_env.task_description, t_max,
                              np.asarray(libero_env._init_states), np.load(path)))
        log.info(f"{len(tasks)} tasks: {[t.label for t in tasks]}")
        return tasks

    def adapter(self, task: Task):
        return self.envs.get(task.suite, task.task_id)

    # ---------------------------------------------------------------- multi-GPU (no-ops on one GPU)
    def _barrier(self) -> None:
        if self.world > 1:
            dist.barrier(group=self.cpu_group)

    def _gather(self, obj) -> list | None:
        """Every rank's obj, in rank order, on rank 0 (None elsewhere)."""
        if self.world == 1:
            return [obj]
        out = [None] * self.world if self.rank == 0 else None
        dist.gather_object(obj, out, dst=0, group=self.cpu_group)
        return out

    def _broadcast(self, obj):
        """Rank 0's obj on every rank."""
        if self.world == 1:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.cpu_group)
        return box[0]

    # ---------------------------------------------------------------- checkpoints
    def _save(self, name: str) -> None:
        if self.rank > 0:
            return
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
            # one state per GPU the saver saw; a torchrun rank sees one GPU, so restore only as many as exist
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"][:torch.cuda.device_count()])

    # ---------------------------------------------------------------- counters derived from the buffer
    @property
    def env_step(self) -> int:
        return sum(ep["steps"] for ep in self.buffer.episodes)

    def axes(self, **extra) -> dict:
        return {"update_step": self.state["update_step"], "env_step": self.env_step, "round": self.state["round"],
                "stage": self.state["stage"], **extra}

    # ---------------------------------------------------------------- rollouts
    def reward(self, rows: dict, t_max) -> tuple[np.ndarray, np.ndarray]:
        """(reward, bootstrap mask) per chunk; t_max is a scalar or one value per row."""
        r = self.cfg.reward
        return chunk_reward(rows["steered"], rows["terminal"], rows["success"], r.steer_kappa, r.failure_penalty,
                            t_max, r.normalize)

    def returns_to_go(self, chunks: dict, t_max: int) -> np.ndarray:
        r, _ = self.reward(chunks, t_max)
        out, g = np.zeros_like(r), 0.0
        for i in reversed(range(len(r))):  # every episode ends terminal (success, failure or timeout)
            g = r[i] + self.cfg.qam.discount * g
            out[i] = g
        return out

    def remaining(self, per_task: list[int], stage: int, rnd: int) -> list[Task]:
        """Episodes still to collect, grouped by task so each env is opened once (resume counts the buffer)."""
        plan = []
        for task, target in zip(self.tasks, per_task, strict=True):
            have = self.buffer.episode_count(stage=stage, round=rnd, suite=task.suite, task=task.task_id)
            plan += [task] * max(0, target - have)
        return plan

    def collect(self, plan: list[Task], *, stage: int, rnd: int, online: bool, use_residual: bool,
                use_edit: bool) -> None:
        """Waves of one episode per rank. Rank r of a wave fills buffer slot len(episodes) + r with plan[done + r],
        so slots, RNG keys and plan entries line up the same for any number of GPUs."""
        for done in range(0, len(plan), self.world):
            slot, mine = len(self.buffer.episodes) + self.rank, None
            if done + self.rank < len(plan):
                mine = self._collect_episode(plan[done + self.rank], slot, online=online, use_residual=use_residual,
                                             use_edit=use_edit)
            results = self._gather(mine)
            if self.rank == 0:
                for result in results:
                    if result is not None:
                        self._record_episode(result, stage=stage, rnd=rnd, online=online)
            self._barrier()
            if self.rank > 0:
                self.buffer.reload()

    def _collect_episode(self, task: Task, slot: int, *, online: bool, use_residual: bool, use_edit: bool) -> dict:
        cfg, errors = self.cfg, []
        while True:
            # Per-episode RNG keyed on the buffer slot: a resumed run continues the sequence instead of replaying
            # the init ids, held-out draws and noise of episodes collected after the last checkpoint.
            episode_rng = np.random.default_rng([cfg.seed, slot, len(errors)])
            torch.manual_seed(int(episode_rng.integers(2 ** 62)))
            init_id = int(episode_rng.integers(len(task.train_inits)))
            start = time.perf_counter()
            try:
                ep = run_episode(self.adapter(task), self.pi05, self.agent, task_idx=task.index,
                                 init_state=task.train_inits[init_id], t_max=task.t_max, use_residual=use_residual,
                                 use_edit=use_edit, record_video=slot % cfg.logging.video_train_every_episodes == 0)
            except Exception:  # an env failure drops this episode, not the run
                errors.append({"suite": task.suite, "task": task.task_id, "init_id": init_id,
                               "error": traceback.format_exc()})
                log.exception(f"Episode failed ({task.label}, train init {init_id}); dropped")
                if len(errors) >= MAX_CONSECUTIVE_ENV_ERRORS:
                    raise
                continue
            video = None
            if ep.frames:
                video = save_video(self.run_dir, f"train_ep{slot + 1}_{task.label}_"
                                   f"{'success' if ep.success else 'fail'}", ep.frames, VIDEO_FPS)
                ep.frames = None
            return {"task": task.index, "init_id": init_id, "ep": ep, "video": video, "errors": errors,
                    "heldout": (not online) and episode_rng.random() < cfg.offline.heldout_frac,
                    "return": float(self.returns_to_go(ep.chunks, task.t_max)[0]),
                    "wall": time.perf_counter() - start}

    def _record_episode(self, result: dict, *, stage: int, rnd: int, online: bool) -> None:
        task, ep, init_id, ret = self.tasks[result["task"]], result["ep"], result["init_id"], result["return"]
        for error in result["errors"]:
            self.monitor.episode({"stage": stage, "round": rnd, **error})
        meta = {"stage": stage, "round": rnd, "suite": task.suite, "task": task.task_id, "init_id": init_id,
                "success": ep.success, "steps": ep.steps, "label_source": LABEL_SOURCE}
        self.buffer.add_episode(ep.chunks, meta, online=online, heldout=result["heldout"])
        self.monitor.episode({**meta, "chunks": len(ep.q), "return": ret, "q": ep.q, "heldout": result["heldout"],
                              "wall_time": result["wall"], **ep.stats})
        self.monitor.log({"episode/success": float(ep.success), "episode/length": ep.steps, "episode/return": ret,
                          "episode/task_index": task.index, "episode/init_id": init_id,
                          "episode/wall_time": result["wall"], **{f"episode/{k}": v for k, v in ep.stats.items()}},
                         **self.axes())
        if result["video"] is not None:
            self.monitor.video(result["video"], **self.axes())

    def evaluate(self, *, use_residual: bool, use_edit: bool, tag: str, episodes_per_task: int) -> float:
        """Official LIBERO-PRO init states, split over ranks; these episodes never enter the buffer."""
        cfg = self.cfg
        items = [(task, i) for task in self.tasks for i in range(episodes_per_task)]
        mine = [self._eval_episode(task, i, use_residual=use_residual, use_edit=use_edit, tag=tag)
                for task, i in items[self.rank::self.world]]
        gathered = self._gather(mine)
        if self.rank > 0:
            return self._broadcast(None)
        results = sorted((r for rank_results in gathered for r in rank_results), key=lambda r: (r["task"], r["i"]))
        rows, successes, qs, rtgs, stats = [], defaultdict(list), [], [], defaultdict(list)
        for r in results:
            task = self.tasks[r["task"]]  # by index: Task hashes by identity and results may be unpickled copies
            if r["ep"] is None:
                rows.append([task.suite, task.task_id, r["init_id"], "error", 0])
                continue
            ep = r["ep"]
            successes[task].append(ep.success)
            rows.append([task.suite, task.task_id, r["init_id"], ep.success, ep.steps])
            qs += ep.q
            rtgs += r["rtg"]
            for k, v in ep.stats.items():
                stats[k].append(v)
            stats["length"].append(ep.steps)
            if r["video"] is not None:
                self.monitor.video(r["video"], **self.axes())
        overall = float(np.mean([s for v in successes.values() for s in v])) if successes else 0.0
        metrics = {f"eval/{tag}/success": overall}
        by_suite = defaultdict(list)
        for task, outcomes in successes.items():
            metrics[f"eval/{tag}/success_{task.label}"] = float(np.mean(outcomes))
            by_suite[task.suite] += outcomes
        metrics.update({f"eval/{tag}/success_{suite}": float(np.mean(v)) for suite, v in by_suite.items()})
        metrics.update({f"eval/{tag}/{k}": float(np.mean(v)) for k, v in stats.items()})
        metrics.update(self._q_vs_return(np.array(qs), np.array(rtgs), f"eval/{tag}"))
        self.monitor.log(metrics, **self.axes())
        self.monitor.table(f"{tag}_by_init", ["suite", "task", "init_id", "success", "steps"], rows, **self.axes())
        per_suite = {suite: round(float(np.mean(v)), 3) for suite, v in by_suite.items()}
        log.info(f"Eval [{tag}] round {self.state['round']} ({episodes_per_task} per task): success {overall:.3f} "
                 f"per suite {per_suite} per task {[(t.label, round(float(np.mean(v)), 2)) for t, v in successes.items()]}")
        return self._broadcast(overall)

    def _eval_episode(self, task: Task, i: int, *, use_residual: bool, use_edit: bool, tag: str) -> dict:
        init_id = i % len(task.official_inits)
        result = {"task": task.index, "i": i, "init_id": init_id, "ep": None, "rtg": [], "video": None}
        try:
            ep = run_episode(self.adapter(task), self.pi05, self.agent, task_idx=task.index,
                             init_state=task.official_inits[init_id], t_max=task.t_max, use_residual=use_residual,
                             use_edit=use_edit, record_video=i < self.cfg.logging.video_eval_per_task)
        except Exception:
            log.exception(f"Eval episode failed ({task.label}, official init {init_id}); excluded")
            return result
        if ep.frames:
            result["video"] = save_video(self.run_dir, f"eval_{tag}_r{self.state['round']}_{task.label}_i{init_id}_"
                                         f"{'success' if ep.success else 'fail'}", ep.frames, VIDEO_FPS)
        result["rtg"] = self.returns_to_go(ep.chunks, task.t_max).tolist()
        ep.chunks, ep.frames = None, None  # only the summary goes back to rank 0
        result["ep"] = ep
        return result

    @staticmethod
    def _q_vs_return(q: np.ndarray, rtg: np.ndarray, prefix: str) -> dict:
        if len(q) < 2 or q.std() == 0 or rtg.std() == 0:
            return {}
        return {f"{prefix}/q_return_corr": float(np.corrcoef(q, rtg)[0, 1]), f"{prefix}/q_bias": float((q - rtg).mean())}

    @torch.no_grad()
    def gate(self) -> None:
        """Spec §10 Stage 2 gate: critic Q of held-out offline chunks vs their realized return-to-go."""
        if self.rank > 0:
            return
        episodes = [ep for ep in self.buffer.episodes if ep["heldout"]]
        if not episodes:
            return
        qs, rtgs, q_success, q_fail = [], [], [], []
        for ep in episodes:
            rows = np.arange(ep["start"], ep["start"] + ep["length"])
            data = self.buffer.get(rows, ["h_prefix", "state_norm", "step", "task", "a_exec", "steered", "terminal",
                                          "success"])
            t_max = int(self.t_max_by_task[data["task"][0]])
            ctx = Context(h_prefix=torch.from_numpy(data["h_prefix"]).float().to(self.cfg.device),
                          state=torch.from_numpy(data["state_norm"]).to(self.cfg.device),
                          progress=torch.from_numpy(data["step"] / t_max).float().to(self.cfg.device), base=None)
            q = self.agent.pessimistic_q(ctx, torch.from_numpy(data["a_exec"]).to(self.cfg.device)).cpu().numpy()
            qs += q.tolist()
            rtgs += self.returns_to_go(data, t_max).tolist()
            (q_success if ep["success"] else q_fail).append(q[0])
        metrics = self._q_vs_return(np.array(qs), np.array(rtgs), "gate")
        metrics.update({"gate/q0_success_episodes": float(np.mean(q_success)) if q_success else float("nan"),
                        "gate/q0_failed_episodes": float(np.mean(q_fail)) if q_fail else float("nan")})
        self.monitor.log(metrics, **self.axes())

    # ---------------------------------------------------------------- updates
    def _context(self, rows: dict) -> Context:
        tasks = [self.tasks[i].text for i in rows["task"]]
        batch = self.pi05.preprocess(rows["image"], rows["image2"], rows["state"], tasks)
        return self.pi05.context(batch, torch.from_numpy(rows["step"] / self.t_max_by_task[rows["task"]]).float())

    def train(self, n_updates: int, offline_ratio: float, stage_ckpt: bool) -> None:
        cfg, device = self.cfg, self.cfg.device
        obs_keys = ["image", "image2", "state", "task", "step"]
        window = defaultdict(list)
        # strided: sample() puts protected rows before online ones, so a contiguous share would give rank 0
        # (which logs batch metrics) only offline data
        mine = slice(self.rank, None, self.world)
        for _ in range(n_updates):
            idx = next_idx = None
            if self.rank == 0:
                idx = self.buffer.sample(cfg.train.batch_size, offline_ratio, self.rng)
                next_idx = self.buffer.next_index(idx)
            idx, next_idx = self._broadcast((idx, next_idx))
            with self.timer("sample"):
                rows = self.buffer.get(idx[mine], obs_keys + ["a_exec", "steered", "terminal", "success"])
                next_rows = self.buffer.get(next_idx[mine], obs_keys)
            with self.timer("prefix_encode"):
                ctx, next_ctx = self._context(rows), self._context(next_rows)
            reward, mask = self.reward(rows, self.t_max_by_task[rows["task"]])
            with self.timer("total_agent"):
                info, hist = self.agent.update(ctx, torch.from_numpy(rows["a_exec"]).to(device),
                                               torch.from_numpy(reward).to(device), torch.from_numpy(mask).to(device),
                                               next_ctx)
            self.state["update_step"] += 1
            step = self.state["update_step"]
            if cfg.logging.nan_guard:
                self._nan_guard(info, idx, next_idx)
            if self.rank > 0:  # info is already averaged over ranks; rank 0 logs and checkpoints
                continue
            self._log_update(window, info, hist, idx, reward, rows["terminal"])
            if stage_ckpt and step % cfg.train.ckpt_every_updates == 0:
                self._save("last.pt")

    def _log_update(self, window: dict, info: dict, hist: dict, idx: np.ndarray, reward: np.ndarray,
                    terminal: np.ndarray) -> None:
        """Accumulate one update's scalars into window; flush averages / histograms on the logging cadence."""
        cfg, step = self.cfg, self.state["update_step"]
        for k, v in info.items():
            v = v.detach().float().cpu()
            if v.numel() == 1:
                window[k].append(v.item())
            else:  # per-flow-step vectors
                for i, x in enumerate(v.tolist()):
                    window[f"{k}/{i}"].append(x)
        window["buffer/batch_protected_frac"].append(float(np.isin(idx, self.buffer.protected_idx).mean()))
        # reward / terminal stats are over rank 0's rows of the batch
        window["buffer/batch_reward_mean"].append(float(reward.mean()))
        window["buffer/batch_terminal_frac"].append(float(terminal.mean()))
        if step % cfg.logging.log_every_updates == 0:
            metrics = {k: float(np.mean(v)) for k, v in window.items()}
            metrics.update({f"update/time/{k}": t / cfg.logging.log_every_updates for k, t in self.timer.pop().items()})
            metrics.update({"update/time/gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                            "buffer/protected_chunks": len(self.buffer.protected_idx),
                            "buffer/online_chunks": len(self.buffer.online_idx), "update/lr": cfg.qam.lr})
            torch.cuda.reset_peak_memory_stats()
            self.monitor.log(metrics, **self.axes())
            window.clear()
        if step % cfg.logging.hist_every_updates == 0:
            for name, values in hist.items():
                self.monitor.histogram(name, values, **self.axes())

    def _nan_guard(self, info: dict, idx: np.ndarray, next_idx: np.ndarray) -> None:
        bad = [k for k, v in info.items() if not torch.isfinite(v).all()]
        if not bad:
            return
        path = self.run_dir / "debug" / f"nan_{self.state['update_step']}.pt"
        if self.rank > 0:  # info is averaged over ranks, so every rank stops here; rank 0 dumps
            raise RuntimeError(f"Non-finite update metrics {bad}; see rank 0's dump {path}")
        path.parent.mkdir(exist_ok=True)
        torch.save({"idx": idx, "next_idx": next_idx, "agent": self.agent.state_dict(),
                    "info": {k: v.detach().cpu() for k, v in info.items()}}, path)
        raise RuntimeError(f"Non-finite update metrics {bad}; batch and agent state saved to {path}")

    # ---------------------------------------------------------------- stages
    def run(self) -> None:
        cfg, s = self.cfg, self.state
        if s["stage"] == 1:
            if cfg.eval.baseline_at_start and not s["baseline_done"]:
                self.evaluate(use_residual=False, use_edit=False, tag="baseline",
                              episodes_per_task=cfg.eval.final_episodes_per_task)
                s["baseline_done"] = True
                self._save("last.pt")
            plan = self.remaining([cfg.offline.episodes_per_task] * len(self.tasks), stage=1, rnd=0)
            self.collect(plan, stage=1, rnd=0, online=False, use_residual=False, use_edit=False)
            s["stage"] = 2
            self._save("last.pt")
        if s["stage"] == 2:
            self.train(cfg.offline.updates - s["update_step"], offline_ratio=1.0, stage_ckpt=True)
            self.gate()
            self._save("stage2.pt")
            self._barrier()  # ranks > 0 load stage2.pt each round when reinit_each_round
            s["stage"] = 3
            self._save("last.pt")
        while s["round"] < cfg.online.rounds:
            rnd = s["round"] + 1
            start = time.perf_counter()
            if cfg.online.reinit_each_round:
                self._load(self.run_dir / "stage2.pt", full=False)
            self.collect(self.remaining(self._round_targets(rnd), stage=3, rnd=rnd), stage=3, rnd=rnd, online=True,
                         use_residual=True, use_edit=True)
            self.train(cfg.online.updates_per_round, cfg.online.offline_ratio, stage_ckpt=False)
            s["round"] = rnd
            self._log_round(rnd, time.perf_counter() - start)
            self.gate()
            final = rnd == cfg.online.rounds
            if rnd % cfg.eval.every_rounds == 0 or final:
                episodes = cfg.eval.final_episodes_per_task if final else cfg.eval.episodes_per_task
                success = self.evaluate(use_residual=True, use_edit=True, tag="policy", episodes_per_task=episodes)
                if success > s["best_eval"]:
                    s["best_eval"] = success
                    self._save("best.pt")
            self._save("last.pt")
        if cfg.online.rounds == 0:  # no online phase (e.g. eval-only with policy.residual_ckpt): evaluate once
            self.evaluate(use_residual=True, use_edit=True, tag="policy",
                          episodes_per_task=cfg.eval.final_episodes_per_task)

    def _round_targets(self, rnd: int) -> list[int]:
        """Split online.episodes_per_round over the tasks; the remainder rotates so every task gets equal data."""
        n, per_round = len(self.tasks), self.cfg.online.episodes_per_round
        base, extra = divmod(per_round, n)
        return [base + int((i - rnd * extra) % n < extra) for i in range(n)]

    def _log_round(self, rnd: int, wall: float) -> None:
        if self.rank > 0:
            return
        this_round = [ep for ep in self.buffer.episodes if ep["stage"] == 3 and ep["round"] == rnd]
        metrics = {"round/wall_time": wall, "round/mean_length": float(np.mean([ep["steps"] for ep in this_round])),
                   "round/train_success": float(np.mean([ep["success"] for ep in this_round]))}
        for suite in {t.suite for t in self.tasks}:
            outcomes = [ep["success"] for ep in this_round if ep["suite"] == suite]
            metrics[f"round/train_success_{suite}"] = float(np.mean(outcomes)) if outcomes else float("nan")
        # A task whose every training episode so far failed gives QAM nothing to learn (identical returns).
        succeeded = {(ep["suite"], ep["task"]) for ep in self.buffer.episodes if ep["success"]}
        never = [t.label for t in self.tasks if (t.suite, t.task_id) not in succeeded]
        metrics["round/never_succeeded_tasks"] = len(never)
        if never:
            log.warning(f"Round {rnd}: {len(never)}/{len(self.tasks)} tasks have no successful training episode yet, "
                        f"so QAM has nothing to learn from them: {never}")
        self.monitor.log(metrics, **self.axes())


@hydra.main(config_path=".", config_name="hyperparameters", version_base="1.3")
def main(cfg) -> None:
    if "LOCAL_RANK" in os.environ:  # torchrun (run_residual.sh -g N): one GPU per rank
        local = int(os.environ["LOCAL_RANK"])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        gpu = visible[local] if visible[0] else str(local)
        # Before CUDA or EGL starts; robosuite's EGL renderer ignores CUDA_VISIBLE_DEVICES without its own id.
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu
        dist.init_process_group("nccl" if cfg.device == "cuda" else "gloo", timeout=DIST_TIMEOUT)
    rank = dist.get_rank() if dist.is_initialized() else 0
    run_dir = Path(cfg.train.resume_dir).resolve() if cfg.train.resume_dir else Path(HydraConfig.get().runtime.output_dir)
    if dist.is_initialized():  # each rank's Hydra timestamp may differ by a second
        box = [run_dir]
        dist.broadcast_object_list(box, src=0)
        run_dir = box[0]
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if rank > 0:  # Hydra's train.log belongs to rank 0
        for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
            root.removeHandler(handler)
    if cfg.train.resume_dir or rank > 0:  # keep one train.log per run dir (and rank) across restarts
        root.addHandler(logging.FileHandler(run_dir / ("train.log" if rank == 0 else f"train_rank{rank}.log")))
    set_seed(cfg.seed)
    monitor = None
    if rank == 0:
        meta_path = run_dir / "meta.json"
        starts = json.loads(meta_path.read_text()) if meta_path.exists() else []
        starts.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
                       "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                       "gpus": dist.get_world_size() if dist.is_initialized() else 1,
                       "python": platform.python_version(), "vls": git_state(VLS_ROOT),
                       "root": git_state(VLS_ROOT.parents[1]), "overrides": list(HydraConfig.get().overrides.task)})
        meta_path.write_text(json.dumps(starts, indent=2))
        monitor = Monitor(cfg, run_dir)
    try:
        Trainer(cfg, run_dir, monitor).run()
    finally:
        if monitor is not None:
            monitor.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

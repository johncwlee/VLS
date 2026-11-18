"""Train a residual on frozen π0.5 on LIBERO-PRO (spec Stages 1-3, no agent yet), or evaluate one (mode=eval).

Baseline: plain π0.5 on seeded eval items (task, init set, init id); every later evaluation is paired with it.
Stage 1: π0.5-only rollouts on seeded train inits -> protected pool (some episodes held out for the gate).
Stage 2: offline.updates on the protected pool (critic only in action_residual mode; 0 by default, so the pool only
  feeds the online batches); critic features are standardized with the pool's statistics first. Gate = critic Q vs
  realized return-to-go on held-out episodes.
Stage 3: rounds of [collect into the online pool; 50:50 updates; gate; paired eval every eval.every_rounds].
  action_residual: collection adds exploration noise. The actor stays at zero for the first actor.warmup_rounds
  rounds (critic warm-up); gate/* is logged only.
  qam_velocity: collection and updates as QAM-E (velocity residual + edit).
Run from the repo root: scripts/run/run_residual.sh [overrides]; resume with train.resume_dir=<run dir>.
Multi-GPU (-g N, torchrun): one rank per GPU, each with its own π0.5 and envs. Collection runs in waves of one
episode per rank, eval splits the episodes over ranks, and each rank trains on train.batch_size rows of an
N x batch_size update with averaged gradients. Rank 0 owns the buffer writes, logging, checkpoints and decisions.
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
import zlib
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
from residual.monitor import Monitor, Timer, paired_stats, save_video
from residual.pi05 import Pi05
from residual.qam_e import Context, QAMEAgent
from residual.rollout import LABEL_SOURCE, VLS_ROOT, EnvPool, run_episode, task_specs

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ENV_ERRORS = 5
VIDEO_FPS = 20
# Ranks wait for each other's longest episode / eval share, and for rank 0's gate and checkpoint I/O.
DIST_TIMEOUT = timedelta(hours=2)
INIT_SETS = ("official", "train")  # position = the init-set id in eval seeds
STORED_CTX_KEYS = ["h_prefix", "state_norm", "task", "step", "a_base"]  # critic/actor inputs without a π0.5 pass
PROGRESS_EVERY_EPISODES = 10


def eval_key(task: "Task", init_set: str, init_id: int, seed: int) -> str:
    """Identifies an eval episode across runs and task subsets (paired evaluation)."""
    return f"{task.suite}/t{task.task_id}/{init_set}/{init_id}/s{seed}"


def eval_seed(seed: int, task: "Task", init_set: str, init_id: int) -> int:
    """Episode seed from stable ids (suite name, LIBERO task id), never from Python's hash() or the run's task order."""
    ids = [seed, zlib.crc32(task.suite.encode()), task.task_id, INIT_SETS.index(init_set), init_id]
    return int(np.random.SeedSequence(ids).generate_state(1)[0])


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
        unknown = set(cfg.eval.init_sets) - set(INIT_SETS)
        if unknown:
            raise SystemExit(f"eval.init_sets: unknown {sorted(unknown)}; expected a subset of {list(INIT_SETS)}")
        if cfg.mode not in ("train", "eval"):
            raise SystemExit(f"mode={cfg.mode!r}; expected train or eval")
        if cfg.mode == "train" and cfg.policy.residual_ckpt:  # training refits the feature statistics the weights need
            raise SystemExit("policy.residual_ckpt is for mode=eval; continue a run with train.resume_dir, reuse "
                             "baseline outcomes with eval.baseline_ckpt")
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
        # baseline: {eval_key: success} of plain π0.5; actor_on: action_residual warm-up is over
        self.state = {"stage": 1, "round": 0, "update_step": 0, "best_eval": -1.0, "baseline": {},
                      "actor_on": False, "warmup_eval_done": False}
        if cfg.policy.residual_ckpt:
            payload = torch.load(cfg.policy.residual_ckpt, weights_only=True)
            self.agent.load_state_dict(payload["agent"])
            self.state["baseline"].update(payload["state"].get("baseline", {}))
        if cfg.eval.baseline_ckpt:  # baseline outcomes only: any actor.bound / architecture can pair with them
            self.state["baseline"].update(torch.load(cfg.eval.baseline_ckpt, weights_only=True)["state"]["baseline"])
        libero_env = self.adapter(self.tasks[0])._env[0]
        specs = field_specs((3, libero_env.observation_height, libero_env.observation_width), self.dims)
        # eval collects nothing
        capacity = 1 if cfg.mode == "eval" else cfg.train.buffer_capacity_per_task * len(self.tasks)
        if self.rank == 0:
            self.buffer = ReplayBuffer(run_dir / "buffer", capacity, specs)
        self._barrier()
        if self.rank > 0:
            self.buffer = ReplayBuffer(run_dir / "buffer", capacity, specs, readonly=True)
        self.eval_summary_path = run_dir / "eval_summary.json"
        self.round_walls: list[float] = []
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

    @property
    def action_mode(self) -> bool:
        return self.agent.actor is not None

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
            # keys added since the checkpoint keep their defaults; baseline outcomes from eval.baseline_ckpt stay
            baseline = {**self.state["baseline"], **payload["state"].get("baseline", {})}
            self.state.update(payload["state"], baseline=baseline)
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
                "stage": self.state["stage"], "chunks": self.buffer.size, **extra}

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

    def collect(self, plan: list[Task], *, stage: int, rnd: int, online: bool) -> None:
        """online: the policy (π0.5 + residual, with exploration noise in action_residual mode) into the online pool;
        otherwise plain π0.5 into the protected pool. Waves of one episode per rank. Rank r of a wave fills buffer slot len(episodes) + r with plan[done + r],
        so slots, RNG keys and plan entries line up the same for any number of GPUs."""
        for done in range(0, len(plan), self.world):
            slot, mine = len(self.buffer.episodes) + self.rank, None
            if done + self.rank < len(plan):
                mine = self._collect_episode(plan[done + self.rank], slot, online)
            results = self._gather(mine)
            if self.rank == 0:
                for result in results:
                    if result is not None:
                        self._record_episode(result, stage=stage, rnd=rnd, online=online)
                n = min(done + self.world, len(plan))
                if n % PROGRESS_EVERY_EPISODES < self.world or n == len(plan):
                    log.info(f"Collect stage {stage} round {rnd}: {n}/{len(plan)} episodes, {self.buffer.size} chunks")
            self._barrier()
            if self.rank > 0:
                self.buffer.reload()

    def _collect_episode(self, task: Task, slot: int, online: bool) -> dict:
        cfg, errors = self.cfg, []
        explore_std = self.agent.explore_std() if online and self.action_mode else None
        while True:
            # Per-episode RNG keyed on the buffer slot: a resumed run continues the sequence instead of replaying
            # the init ids, held-out draws and noise of episodes collected after the last checkpoint.
            episode_rng = np.random.default_rng([cfg.seed, slot, len(errors)])
            torch.manual_seed(int(episode_rng.integers(2 ** 62)))
            init_id = int(episode_rng.integers(len(task.train_inits)))
            start = time.perf_counter()
            try:
                ep = run_episode(self.adapter(task), self.pi05, self.agent, task_idx=task.index,
                                 init_state=task.train_inits[init_id], t_max=task.t_max, use_residual=online,
                                 record_video=slot % cfg.logging.video_train_every_episodes == 0,
                                 explore_std=explore_std)
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
                    "heldout": episode_rng.random() < (cfg.online if online else cfg.offline).heldout_frac,
                    "return": float(self.returns_to_go(ep.chunks, task.t_max)[0]),
                    "wall": time.perf_counter() - start}

    def import_expert(self) -> None:
        """offline.source=vls: the successful episodes VLS recorded for this run's tasks (main.py, main.record_dir)
        are the protected pool, as off-policy chunks. Features are recomputed from the stored images;
        noise and latent stay zero (VLS does not produce them; the action residual never reads them). Episodes
        carry stage 0, so the Stage 1 plan does not count them, and a resumed run skips the files it has."""
        cfg = self.cfg
        for task in self.tasks if self.rank == 0 else []:
            task_dir = Path(cfg.offline.expert_dir) / task.suite / f"task{task.task_id}"
            for path in sorted(task_dir.glob("*_success.npz")):
                if self.buffer.episode_count(expert=path.name, suite=task.suite, task=task.task_id):
                    continue
                data = np.load(path)
                n = len(data["a_exec"])
                chunks = {k: data[k] for k in ("image", "image2", "state", "steered", "terminal", "success")}
                chunks.update(task=np.full(n, task.index, dtype=np.int16), step=np.arange(n, dtype=np.int16),
                              trigger=np.zeros(n, dtype=np.uint8))
                ctx = self._context(chunks)
                chunks.update({k: self.agent.clip(torch.from_numpy(data[k]).to(cfg.device)).cpu().numpy()
                               for k in ("a_exec", "a_base")},
                              h_prefix=ctx.h_prefix.float().cpu().numpy().astype(np.float16),
                              state_norm=ctx.state.cpu().numpy())
                slot = len(self.buffer.episodes)
                meta = {"stage": 0, "round": 0, "suite": task.suite, "task": task.task_id, "expert": path.name,
                        "init_id": int(data["init_id"]), "success": True, "steps": int(data["steps"]),
                        "label_source": LABEL_SOURCE}
                self.buffer.add_episode(chunks, meta, online=False, heldout=bool(
                    np.random.default_rng([cfg.seed, slot]).random() < cfg.offline.heldout_frac))
        if self.rank == 0:
            if not self.buffer.size:
                raise SystemExit("offline.source=vls: no *_success.npz of this run's tasks under "
                                 f"{cfg.offline.expert_dir}")
            log.info(f"Offline pool: {self.buffer.episode_count(stage=0)} VLS episodes, {self.buffer.size} chunks")
        self._barrier()
        if self.rank > 0:
            self.buffer.reload()

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

    # ---------------------------------------------------------------- evaluation
    def eval_items(self, init_set: str, per_task: int) -> list[tuple[int, str, int]]:
        """(task index, init set, init id) for the first per_task inits of every task."""
        items = []
        for task in self.tasks:
            n = len(task.official_inits if init_set == "official" else task.train_inits)
            items += [(task.index, init_set, i) for i in range(min(per_task, n))]
        return items

    @staticmethod
    def _tag(name: str, init_set: str) -> str:
        return name if init_set == "official" else f"{name}_{init_set}"

    def evaluate(self, items: list[tuple[int, str, int]], *, tag: str, use_residual: bool,
                 explore_std: torch.Tensor | None = None) -> dict[str, bool]:
        """Seeded episodes on (task index, init set, init id) items, split over ranks; never added to the buffer.
        Returns {eval_key: success} on every rank (episodes that raised are left out)."""
        items = self._broadcast(items)
        mine, results = items[self.rank::self.world], []
        for k, (task_index, init_set, init_id) in enumerate(mine, 1):
            results.append(self._eval_episode(self.tasks[task_index], init_set, init_id, tag=tag,
                                              use_residual=use_residual, explore_std=explore_std))
            if self.rank == 0 and (k % PROGRESS_EVERY_EPISODES == 0 or k == len(mine)):
                log.info(f"Eval [{tag}]: {k}/{len(mine)} episodes on rank 0 ({len(items)} over {self.world} GPU(s))")
        gathered = self._gather(results)
        outcomes = self._log_eval(tag, [r for rank_results in gathered for r in rank_results]) if self.rank == 0 else None
        return self._broadcast(outcomes)

    def _log_eval(self, tag: str, results: list[dict]) -> dict[str, bool]:
        """Rank 0: success per task / suite, episode stats, Q vs return, the per-init table; returns the outcomes."""
        results = sorted(results, key=lambda r: (r["task"], r["init_set"], r["init_id"]))
        rows, successes, qs, rtgs, stats, outcomes = [], defaultdict(list), [], [], defaultdict(list), {}
        for r in results:
            task = self.tasks[r["task"]]  # by index: Task hashes by identity and results may be unpickled copies
            if r["ep"] is None:
                rows.append([task.suite, task.task_id, r["init_set"], r["init_id"], "error", 0])
                continue
            ep = r["ep"]
            outcomes[r["key"]] = ep.success
            successes[task].append(ep.success)
            rows.append([task.suite, task.task_id, r["init_set"], r["init_id"], ep.success, ep.steps])
            qs += ep.q
            rtgs += r["rtg"]
            for k, v in ep.stats.items():
                stats[k].append(v)
            stats["length"].append(ep.steps)
            if r["video"] is not None:
                self.monitor.video(r["video"], **self.axes())
        overall = float(np.mean([s for v in successes.values() for s in v])) if successes else 0.0
        metrics = {f"eval/{tag}/success": overall, f"eval/{tag}/errors": len(results) - len(outcomes)}
        by_suite = defaultdict(list)
        for task, task_successes in successes.items():
            metrics[f"eval/{tag}/success_{task.label}"] = float(np.mean(task_successes))
            by_suite[task.suite] += task_successes
        metrics.update({f"eval/{tag}/success_{suite}": float(np.mean(v)) for suite, v in by_suite.items()})
        metrics.update({f"eval/{tag}/{k}": float(np.mean(v)) for k, v in stats.items()})
        metrics.update(self._q_vs_return(np.array(qs), np.array(rtgs), f"eval/{tag}"))
        self.monitor.log(metrics, **self.axes())
        self.monitor.table(f"{tag}_by_init", ["suite", "task", "init_set", "init_id", "success", "steps"], rows,
                           **self.axes())
        per_suite = {suite: round(float(np.mean(v)), 3) for suite, v in by_suite.items()}
        log.info(f"Eval [{tag}] round {self.state['round']} ({len(results)} episodes): success {overall:.3f} "
                 f"per suite {per_suite} per task {[(t.label, round(float(np.mean(v)), 2)) for t, v in successes.items()]}")
        return outcomes

    def _eval_episode(self, task: Task, init_set: str, init_id: int, *, tag: str, use_residual: bool,
                      explore_std: torch.Tensor | None) -> dict:
        inits = task.official_inits if init_set == "official" else task.train_inits
        seed = self.cfg.eval.seed
        result = {"task": task.index, "init_set": init_set, "init_id": init_id, "ep": None, "rtg": [], "video": None,
                  "key": eval_key(task, init_set, init_id, seed)}
        try:
            ep = run_episode(self.adapter(task), self.pi05, self.agent, task_idx=task.index,
                             init_state=inits[init_id], t_max=task.t_max, use_residual=use_residual,
                             record_video=init_id < self.cfg.logging.video_eval_per_task,
                             seed=eval_seed(seed, task, init_set, init_id), explore_std=explore_std)
        except Exception:
            log.exception(f"Eval episode failed ({task.label}, {init_set} init {init_id}); excluded")
            return result
        if ep.frames:
            result["video"] = save_video(self.run_dir, f"eval_{tag}_r{self.state['round']}_{task.label}_{init_set}"
                                         f"{init_id}_{'success' if ep.success else 'fail'}", ep.frames, VIDEO_FPS)
        result["rtg"] = self.returns_to_go(ep.chunks, task.t_max).tolist()
        ep.chunks, ep.frames = None, None  # only the summary goes back to rank 0
        result["ep"] = ep
        return result

    def _ensure_baseline(self, per_task: int) -> None:
        """Plain π0.5 on every eval item without a stored outcome (for this eval.seed). The outcomes live in the
        checkpoint state, so every later policy evaluation, in this run or a mode=eval run, is paired with them."""
        for init_set in self.cfg.eval.init_sets:
            items = [it for it in self.eval_items(init_set, per_task)
                     if eval_key(self.tasks[it[0]], it[1], it[2], self.cfg.eval.seed) not in self.state["baseline"]]
            if items:
                self.state["baseline"].update(self.evaluate(items, tag=self._tag("baseline", init_set),
                                                            use_residual=False))
                self._save("last.pt")  # per set: a crash or job timeout keeps the finished sets

    def _evaluate_policy(self, per_task: int, zero_residual: bool = False) -> float:
        """The current policy on every init set, paired with the baseline. Returns the success that selects best.pt:
        on the train inits when they are evaluated (the official inits are the test set), over the inits every
        evaluation shares (the final one runs more)."""
        success = None
        for init_set in self.cfg.eval.init_sets:
            tag = self._tag("policy", init_set)
            outcomes = self.evaluate(self.eval_items(init_set, per_task), tag=tag, use_residual=True)
            self._paired_report(tag, outcomes, zero_residual)
            if init_set == "train" or success is None:
                shared = [v for k, v in outcomes.items() if int(k.split("/")[3]) < self.cfg.eval.episodes_per_task]
                success = float(np.mean(shared)) if shared else 0.0
        return success

    def _paired_report(self, tag: str, outcomes: dict[str, bool], zero_residual: bool = False) -> None:
        """Rank 0: paired difference from the baseline, overall and per task -> metrics and eval_summary.json.
        zero_residual: the policy is π0.5 itself, so any flipped outcome is noise in the pairing."""
        if self.rank > 0:
            return
        overall = paired_stats(outcomes, self.state["baseline"])
        if not overall:
            log.warning(f"Eval [{tag}]: no baseline outcome to pair with (eval.baseline_at_start=false?)")
            return
        by_task = defaultdict(dict)
        for key, success in outcomes.items():
            suite, task_id = key.split("/")[:2]
            by_task[f"{suite}_{task_id}"][key] = success
        per_task = {label: paired_stats(o, self.state["baseline"]) for label, o in sorted(by_task.items())}
        metrics = {f"eval/{tag}/{k}": v for k, v in overall.items() if k != "success"}  # success: _log_eval
        metrics.update({f"eval/{tag}/{k}_{label}": v for label, s in per_task.items() for k, v in s.items()
                        if k in ("paired_diff", "paired_se")})
        self.monitor.log(metrics, **self.axes())
        log.info(f"Eval [{tag}] vs baseline over {overall['paired_n']} paired episodes: success "
                 f"{overall['success']:.3f} vs {overall['baseline_success']:.3f}, diff {overall['paired_diff']:+.3f} "
                 f"± {overall['paired_se']:.3f} (SE), flipped {overall['flip_frac']:.3f}")
        if zero_residual and overall["flip_frac"] > 0:
            log.warning(f"Eval [{tag}]: the zero residual differs from the stored baseline on {overall['flip_frac']:.1%} "
                        f"of episodes. The baseline was not produced under this run's conditions (another machine, "
                        f"code version or eval.baseline_ckpt), so every paired difference carries that noise")
        summary = json.loads(self.eval_summary_path.read_text()) if self.eval_summary_path.exists() else {}
        summary[f"{tag}_round{self.state['round']}"] = {**overall, "per_task": per_task}
        tmp = self.eval_summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        os.replace(tmp, self.eval_summary_path)

    @staticmethod
    def _q_vs_return(q: np.ndarray, rtg: np.ndarray, prefix: str) -> dict:
        if len(q) < 2 or q.std() == 0 or rtg.std() == 0:
            return {}
        return {f"{prefix}/q_return_corr": float(np.corrcoef(q, rtg)[0, 1]), f"{prefix}/q_bias": float((q - rtg).mean())}

    def _heldout(self, online: bool) -> tuple[dict, np.ndarray, list[dict], np.ndarray] | None:
        """Every chunk of the held-out offline or online episodes: (rows, return-to-go, episodes, episode starts)."""
        episodes = [ep for ep in self.buffer.episodes if ep["heldout"] and ep["online"] == online]
        if not episodes:
            return None
        rows = np.concatenate([np.arange(ep["start"], ep["start"] + ep["length"]) for ep in episodes])
        data = self.buffer.get(rows, STORED_CTX_KEYS + ["a_exec", "steered", "terminal", "success"])
        starts = np.cumsum([0] + [ep["length"] for ep in episodes])[:-1]
        rtg = np.concatenate([
            self.returns_to_go({k: data[k][a:a + ep["length"]] for k in ("steered", "terminal", "success")},
                               int(self.t_max_by_task[data["task"][a]]))
            for a, ep in zip(starts, episodes, strict=True)])
        return data, rtg, episodes, starts

    @torch.no_grad()
    def gate(self) -> None:
        """Rank 0: gate/* diagnostics on held-out episodes; they are logged and decide nothing.
        Offline episodes: critic Q vs realized return-to-go. A critic that ignores actions scores well there, so on
        online episodes (collected with exploration noise) agent.action_metrics asks whether Q changes with the
        executed action."""
        if self.rank > 0:
            return
        metrics, device = {}, self.cfg.device
        # ponytail: whole held-out set in one batch (~27k rows by round 100 at the defaults); chunk it if the gate
        # runs out of GPU memory
        offline = self._heldout(online=False)
        if offline is not None:
            data, rtg, episodes, starts = offline
            q = self.agent.pessimistic_q(self._stored_context(data),
                                         torch.from_numpy(data["a_exec"]).to(device)).cpu().numpy()
            metrics.update(self._q_vs_return(q, rtg, "gate"))
            q0 = {True: [], False: []}
            for a, ep in zip(starts, episodes, strict=True):
                q0[ep["success"]].append(q[a])
            metrics.update({"gate/q0_success_episodes": float(np.mean(q0[True])) if q0[True] else float("nan"),
                            "gate/q0_failed_episodes": float(np.mean(q0[False])) if q0[False] else float("nan")})
        online = self._heldout(online=True)
        if online is not None:
            data, rtg = online[:2]
            action = self.agent.action_metrics(self._stored_context(data), torch.from_numpy(data["a_exec"]).to(device),
                                               torch.from_numpy(rtg).to(device))
            metrics.update({k: float(v) for k, v in action.items()})
        if metrics:
            self.monitor.log(metrics, **self.axes())

    # ---------------------------------------------------------------- updates
    def _stored_context(self, rows: dict) -> Context:
        """Critic / actor inputs from stored buffer fields, without a π0.5 pass."""
        device = self.cfg.device
        return Context(h_prefix=torch.from_numpy(rows["h_prefix"]).float().to(device),
                       state=torch.from_numpy(rows["state_norm"]).to(device),
                       progress=torch.from_numpy(rows["step"] / self.t_max_by_task[rows["task"]]).float().to(device),
                       base=None, a_base=torch.from_numpy(rows["a_base"]).to(device))

    def _fit_feat_stats(self) -> None:
        """Standardize critic / actor features with the offline pool's statistics; rank 0 computes, all ranks get
        the same values."""
        if self.rank == 0:
            idx = self.buffer.protected_idx
            if len(idx):
                rows = self.buffer.get(idx, STORED_CTX_KEYS)
                self.agent.fit_feat_stats(self.agent.raw_feat(self._stored_context(rows)))
                log.info(f"Critic features standardized with {len(idx)} offline chunks")
            else:
                log.warning("No offline chunks: critic features stay unstandardized")
        if self.world > 1:
            for t in (self.agent.feat_mean, self.agent.feat_std):
                dist.broadcast(t, src=0)

    def _context(self, rows: dict) -> Context:
        tasks = [self.tasks[i].text for i in rows["task"]]
        batch = self.pi05.preprocess(rows["image"], rows["image2"], rows["state"], tasks)
        return self.pi05.context(batch, torch.from_numpy(rows["step"] / self.t_max_by_task[rows["task"]]).float())

    def train(self, n_updates: int, offline_ratio: float, stage_ckpt: bool) -> None:
        cfg, device = self.cfg, self.cfg.device
        # action_residual: stored features and π0.5's chunk a_base, no π0.5 pass; qam_velocity re-encodes images
        stored = self.action_mode
        obs_keys = STORED_CTX_KEYS if stored else ["image", "image2", "state", "task", "step"]
        window = defaultdict(list)
        # strided: sample() puts protected rows before online ones, so a contiguous share would give rank 0
        # (which logs batch metrics) only offline data
        mine = slice(self.rank, None, self.world)
        for _ in range(n_updates):
            idx = next_idx = None
            if self.rank == 0:  # train.batch_size is per GPU
                idx = self.buffer.sample(cfg.train.batch_size * self.world, offline_ratio, self.rng)
                next_idx = self.buffer.next_index(idx)
            idx, next_idx = self._broadcast((idx, next_idx))
            with self.timer("sample"):
                rows = self.buffer.get(idx[mine], obs_keys + ["a_exec", "steered", "terminal", "success"])
                next_rows = self.buffer.get(next_idx[mine], obs_keys)
            if stored:
                ctx, next_ctx = self._stored_context(rows), self._stored_context(next_rows)
            else:
                with self.timer("prefix_encode"):
                    ctx, next_ctx = self._context(rows), self._context(next_rows)
            reward, mask = self.reward(rows, self.t_max_by_task[rows["task"]])
            with self.timer("total_agent"):
                info, hist = self.agent.update(ctx, torch.from_numpy(rows["a_exec"]).to(device),
                                               torch.from_numpy(reward).to(device), torch.from_numpy(mask).to(device),
                                               next_ctx, actor=self.state["actor_on"])
            self.state["update_step"] += 1
            step = self.state["update_step"]
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
                            "buffer/online_chunks": len(self.buffer.online_idx)})
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
        if self.cfg.mode == "eval":
            self.run_eval()
            return
        cfg, s = self.cfg, self.state
        if cfg.eval.baseline_at_start:  # also on resume: only items without a stored outcome are run
            self._ensure_baseline(cfg.eval.final_episodes_per_task)
        if s["stage"] == 1:
            if cfg.offline.source == "vls":
                self.import_expert()
            elif cfg.offline.source == "pi05":
                plan = self.remaining([cfg.offline.episodes_per_task] * len(self.tasks), stage=1, rnd=0)
                self.collect(plan, stage=1, rnd=0, online=False)
            else:
                raise SystemExit(f"offline.source must be 'pi05' or 'vls', got {cfg.offline.source!r}")
            self._fit_feat_stats()
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
            self._round(s["round"] + 1)

    def _round(self, rnd: int) -> None:
        """One online round. The actor flag decides which parameters get gradients, and the all-reduce must match
        across ranks: it depends only on the round number."""
        cfg, s = self.cfg, self.state
        s["round"] = rnd  # for the logged axes; saved only with best.pt / last.pt below
        warmup = self.action_mode and rnd <= cfg.actor.warmup_rounds
        s["actor_on"] = self.action_mode and not warmup
        start = time.perf_counter()
        if cfg.online.reinit_each_round:
            self._load(self.run_dir / "stage2.pt", full=False)
        per_task = [cfg.online.episodes_per_task] * len(self.tasks)
        self.collect(self.remaining(per_task, stage=3, rnd=rnd), stage=3, rnd=rnd, online=True)
        self.train(cfg.online.updates_per_round, cfg.online.offline_ratio, stage_ckpt=False)
        self.gate()
        final = rnd == cfg.online.rounds
        # While the actor is zero the policy is π0.5 itself: one eval is enough. It checks the pairing: it matches
        # a baseline from this run episode for episode (_paired_report warns if not).
        if (rnd % cfg.eval.every_rounds == 0 or final) and not (warmup and s["warmup_eval_done"]):
            s["warmup_eval_done"] |= warmup
            success = self._evaluate_policy(cfg.eval.final_episodes_per_task if final else cfg.eval.episodes_per_task,
                                            zero_residual=warmup)
            if success > s["best_eval"]:
                s["best_eval"] = success
                self._save("best.pt")
        self._log_round(rnd, time.perf_counter() - start)
        self._save("last.pt")

    def run_eval(self) -> None:
        """mode=eval: π0.5 baseline on items without a stored outcome, then policy.residual_ckpt and each
        eval.explore_bounds entry (π0.5 + exploration noise at that s), each paired with the baseline."""
        cfg, per_task = self.cfg, self.cfg.eval.final_episodes_per_task
        if cfg.eval.explore_bounds and not self.action_mode:
            raise SystemExit("eval.explore_bounds needs actor.mode=action_residual")
        self._ensure_baseline(per_task)
        if cfg.policy.residual_ckpt:
            self._evaluate_policy(per_task)
        for bound in cfg.eval.explore_bounds:
            for init_set in cfg.eval.init_sets:
                tag = self._tag(f"explore_s{round(bound * 100):03d}", init_set)  # no dots in wandb keys
                outcomes = self.evaluate(self.eval_items(init_set, per_task), tag=tag, use_residual=False,
                                         explore_std=self.agent.explore_std(bound))
                self._paired_report(tag, outcomes)

    def _log_round(self, rnd: int, wall: float) -> None:
        if self.rank > 0:
            return
        this_round = [ep for ep in self.buffer.episodes if ep["stage"] == 3 and ep["round"] == rnd]
        metrics = {"round/wall_time": wall, "round/mean_length": float(np.mean([ep["steps"] for ep in this_round])),
                   "round/train_success": float(np.mean([ep["success"] for ep in this_round]))}
        for suite in {t.suite for t in self.tasks}:
            outcomes = [ep["success"] for ep in this_round if ep["suite"] == suite]
            metrics[f"round/train_success_{suite}"] = float(np.mean(outcomes)) if outcomes else float("nan")
        # A task whose every training episode so far failed gives nothing to learn (identical returns).
        succeeded = {(ep["suite"], ep["task"]) for ep in self.buffer.episodes if ep["success"]}
        never = [t.label for t in self.tasks if (t.suite, t.task_id) not in succeeded]
        metrics["round/never_succeeded_tasks"] = len(never)
        if never:
            log.warning(f"Round {rnd}: {len(never)}/{len(self.tasks)} tasks have no successful training episode yet, "
                        f"so there is nothing to learn from them: {never}")
        self.round_walls.append(wall)
        rounds = self.cfg.online.rounds
        eta = float(np.mean(self.round_walls)) * (rounds - rnd) / 3600
        actor = "-" if not self.action_mode else "on" if self.state["actor_on"] else "off (warm-up)"
        metrics.update({"progress/chunks": self.buffer.size, "progress/frac": rnd / rounds,
                        "progress/eta_hours": eta, "progress/actor_on": float(self.state["actor_on"])})
        log.info(f"Round {rnd}/{rounds} | {self.buffer.size} chunks | actor {actor} | train success "
                 f"{metrics['round/train_success']:.3f} | round {wall / 60:.1f} min | ETA {eta:.1f} h")
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
        gpus = dist.get_world_size() if dist.is_initialized() else 1
        run_facts = {"gpus": gpus, "effective_batch_size": cfg.train.batch_size * gpus}
        log.info(f"Run dir: {run_dir} | mode {cfg.mode} | actor.mode {cfg.actor.mode} | batch {cfg.train.batch_size} "
                 f"per GPU x {gpus} GPU(s) = {run_facts['effective_batch_size']} per update")
        meta_path = run_dir / "meta.json"
        starts = json.loads(meta_path.read_text()) if meta_path.exists() else []
        starts.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
                       "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, **run_facts,
                       "python": platform.python_version(), "vls": git_state(VLS_ROOT),
                       "root": git_state(VLS_ROOT.parents[1]), "overrides": list(HydraConfig.get().overrides.task)})
        meta_path.write_text(json.dumps(starts, indent=2))
        monitor = Monitor(cfg, run_dir, extra_config=run_facts)
    try:
        Trainer(cfg, run_dir, monitor).run()
    finally:
        if monitor is not None:
            monitor.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

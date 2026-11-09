"""Seeded train init states for the LIBERO-PRO split (spec §11.2), plus the overlap check against the official 50.

Same procedure as LIBERO-PRO's notebooks/generate_init_states.py (sample placements on reset, store the sim state
with no settling) but seeded: state i uses seed `split.seed + i`. It covers the same tasks as training (env.* in
hyperparameters.yaml: one perturbation type across suites), runs each task in its own process (split.workers at a
time, split.attempts tries each), skips tasks whose file already exists, drops layouts that duplicate an official
one, and never touches the official init files.
Run from modules/VLS:
    ../../envs/vls/bin/python -m residual.make_train_inits                   # env.perturbation=task, all suites
    ../../envs/vls/bin/python -m residual.make_train_inits env.perturbation=swap
Reads object poses from the simulator: evaluation-side data-split analysis only, never a policy input.
"""

import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from core.env_adapters.libero_adapter import _apply_perturbations, _get_suite, get_task_init_states
from residual.rollout import task_specs

log = logging.getLogger(__name__)


def object_poses(env, states: np.ndarray) -> np.ndarray:
    """[N, objects, 7] free-joint pose (xyz + wxyz quaternion) of every movable object after loading each state."""
    objects = [o for o in env.env.objects_dict.values() if o.joints]
    poses = []
    for state in states:
        env.set_init_state(state)
        poses.append([env.sim.data.get_joint_qpos(o.joints[-1]).copy() for o in objects])
    return np.asarray(poses)


def pairwise_max_distance(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max over objects of position (m) and rotation (deg) distance for every pair: two [len(a), len(b)] arrays."""
    pos = np.linalg.norm(a[:, None, :, :3] - b[None, :, :, :3], axis=-1).max(-1)
    dot = np.abs((a[:, None, :, 3:] * b[None, :, :, 3:]).sum(-1)).clip(max=1.0)
    rot = np.degrees(2 * np.arccos(dot)).max(-1)
    return pos, rot


def overlap_report(train: np.ndarray, test: np.ndarray, min_pos: float, min_rot: float) -> tuple[dict, np.ndarray]:
    """Returns (report, duplicate mask over train layouts)."""
    pos, rot = pairwise_max_distance(train, test)
    duplicates = ((pos < min_pos) & (rot < min_rot)).any(1)
    nearest = pos.argmin(1)
    nn_pos, nn_rot = pos.min(1), rot[np.arange(len(train)), nearest]
    test_pos, _ = pairwise_max_distance(test, test)
    np.fill_diagonal(test_pos, np.inf)
    quantiles = {"min": 0, "p1": 1, "p5": 5, "median": 50}
    return {"num_train": len(train), "num_test": len(test), "duplicates": int(duplicates.sum()),
            "nn_pos_m": {k: float(np.percentile(nn_pos, q)) for k, q in quantiles.items()},
            "nn_rot_deg": {k: float(np.percentile(nn_rot, q)) for k, q in quantiles.items()},
            "official_vs_official_nn_pos_m": {k: float(np.percentile(test_pos.min(1), q)) for k, q in quantiles.items()}
            }, duplicates


def generate(suite: str, task_id: int, split) -> dict:
    """One task: LIBERO-PRO's procedure (every reset reloads the model and re-samples placements) on a camera-less
    env, so the process never creates an EGL context. Rendering envs rebuilt their renderer on every reset, grew
    worker memory by GBs and crashed workers; the camera-less env gives bit-identical states (checked on
    libero_spatial_task 3 and libero_10_task 0), 0.3-0.6 s per layout. A soft reset (hard_reset=False) would be
    ~140x faster but places objects that sit on fixtures from the fixture's stale pose (libero_spatial_task 3: a
    bowl on the cabinet lands 0.9 m off)."""
    actual_suite, _ = _apply_perturbations(suite)
    task_suite = _get_suite(actual_suite)
    task = task_suite.get_task(task_id)
    out_dir = Path(split.train_init_dir).resolve() / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task.name}.npy"
    if path.exists():
        return {"suite": suite, "task_id": task_id, "skipped": str(path)}
    official = np.asarray(get_task_init_states(task_suite, task_id))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(bddl_file_name=str(bddl), has_offscreen_renderer=False, use_camera_obs=False)
    states = []
    for i in range(split.num_train_inits):
        env.seed(split.seed + i)
        env.reset()
        states.append(env.get_sim_state())
    states = np.stack(states)
    report, duplicates = overlap_report(object_poses(env, states), object_poses(env, official),
                                        split.min_pos_dist_m, split.min_rot_dist_deg)
    # A layout whose every object lies within the thresholds of an official test layout would be "the same
    # episode" as a test one: drop it from training (the file then holds slightly fewer than num_train_inits).
    report.update(suite=suite, task_id=task_id, task=task.name, seed=split.seed, kept=int((~duplicates).sum()),
                  dropped_seeds=[split.seed + int(i) for i in np.flatnonzero(duplicates)])
    (out_dir / f"{task.name}.overlap.json").write_text(json.dumps(report, indent=2))
    np.save(path, states[~duplicates])
    return report


def run_isolated(suite: str, task_id: int, overrides: list[str], log_dir: Path, attempts: int) -> str | None:
    """Generate one task in its own process (a crash only costs this task), retrying; None on success."""
    log_path = log_dir / f"{suite}_t{task_id}.log"
    cmd = [sys.executable, "-m", "residual.make_train_inits", *overrides, f"split.only={suite}:{task_id}",
           f"hydra.run.dir={log_dir / f'{suite}_t{task_id}'}"]
    for attempt in range(1, attempts + 1):
        with log_path.open("a") as out:
            code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
        if code == 0:
            return None
        log.warning(f"{suite} task {task_id}: attempt {attempt}/{attempts} exited with {code} (see {log_path})")
    return f"{suite} task {task_id} (exit {code}, {log_path})"


@hydra.main(config_path=".", config_name="hyperparameters", version_base="1.3")
def main(cfg) -> None:
    split = cfg.split
    if split.only:  # child process: one task
        suite, task_id = split.only.rsplit(":", 1)
        log.info(json.dumps(generate(suite, int(task_id), split)))
        return
    specs = task_specs(cfg.env)
    overrides = [o for o in HydraConfig.get().overrides.task if not o.startswith(("hydra.", "split.only"))]
    log_dir = Path(HydraConfig.get().runtime.output_dir) / "tasks"
    log_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=split.workers) as pool:
        failures = [f for f in pool.map(lambda st: run_isolated(*st, overrides, log_dir, split.attempts), specs) if f]
    if failures:
        raise SystemExit(f"{len(failures)} tasks failed (rerun the same command; finished tasks are skipped):\n"
                         + "\n".join(failures))
    log.info(f"Done: {len(specs)} tasks in {Path(split.train_init_dir).resolve()}")


if __name__ == "__main__":
    main()

"""Seeded train init states for the LIBERO-PRO split (spec §11.2), plus the overlap check against the official 50.

Same procedure as LIBERO-PRO's notebooks/generate_init_states.py (sample placements on reset, store the sim state
with no settling) but seeded: state i uses seed `split.seed + i`. It covers the same tasks as training (env.* in
hyperparameters.yaml: one perturbation type across suites), runs split.workers tasks in parallel, skips tasks
whose file already exists, and never touches the official init files. Run from modules/VLS:
    ../../envs/vls/bin/python -m residual.make_train_inits                   # env.perturbation=task, all suites
    ../../envs/vls/bin/python -m residual.make_train_inits env.perturbation=swap
Reads object poses from the simulator: evaluation-side data-split analysis only, never a policy input.
"""

import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
from libero.libero.envs.env_wrapper import ControlEnv
from omegaconf import OmegaConf

from residual.rollout import make_adapter, task_specs

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


def overlap_report(train: np.ndarray, test: np.ndarray, min_pos: float, min_rot: float) -> dict:
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
            "official_vs_official_nn_pos_m": {k: float(np.percentile(test_pos.min(1), q)) for k, q in quantiles.items()}}


def generate(suite: str, task_id: int, split: dict) -> dict:
    """One task: LIBERO-PRO's procedure (every reset reloads the model and re-samples placements), on a camera-less
    env. The rendering env rebuilt its EGL renderer on every reset, grew ~1-3 GB per worker and segfaulted workers;
    the camera-less env gives bit-identical states (checked on libero_spatial_task 3 and libero_10_task 0) at
    0.3-0.6 s per layout. A soft reset (hard_reset=False) would be ~140x faster but places objects that sit on
    fixtures from the fixture's stale pose (libero_spatial_task 3: a bowl on the cabinet lands 0.9 m off)."""
    libero_env = make_adapter({"suite_name": suite}, task_id)._env[0]  # suite mapping, task name, official inits
    out_dir = Path(split["train_init_dir"]) / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{libero_env.task}.npy"
    if path.exists() and len(np.load(path)) == split["num_train_inits"]:
        return {"suite": suite, "task_id": task_id, "skipped": str(path)}
    official = np.asarray(libero_env._init_states)
    env = ControlEnv(bddl_file_name=libero_env._env.env.bddl_file_name, has_offscreen_renderer=False,
                     use_camera_obs=False)
    states = []
    for i in range(split["num_train_inits"]):
        env.seed(split["seed"] + i)
        env.reset()
        states.append(env.get_sim_state())
    states = np.stack(states)
    report = overlap_report(object_poses(env, states), object_poses(env, official),
                            split["min_pos_dist_m"], split["min_rot_dist_deg"])
    report.update(suite=suite, task_id=task_id, task=libero_env.task, seed=split["seed"])
    (out_dir / f"{libero_env.task}.overlap.json").write_text(json.dumps(report, indent=2))
    if not report["duplicates"]:
        np.save(path, states)
    return report  # no env.close(): the worker process exits after one task


@hydra.main(config_path=".", config_name="hyperparameters", version_base="1.3")
def main(cfg) -> None:
    split = OmegaConf.to_container(cfg.split, resolve=True)
    split["train_init_dir"] = str(Path(split["train_init_dir"]).resolve())
    specs = task_specs(cfg.env)
    duplicated, crashed = [], []
    # A fresh process per task: reusing a worker after its EGL renderer is torn down segfaulted (2026-11-08), and
    # hard resets grow a worker's memory. A crash only loses that task; rerunning skips finished tasks.
    with ProcessPoolExecutor(max_workers=split["workers"], mp_context=multiprocessing.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(generate, suite, task_id, split): (suite, task_id) for suite, task_id in specs}
        for future in as_completed(futures):
            try:
                report = future.result()
            except Exception as error:  # noqa: BLE001 - a crashed worker must not stop the other tasks
                crashed.append(futures[future])
                log.error(f"{futures[future]} failed: {error!r}")
                continue
            log.info(json.dumps(report))
            if report.get("duplicates"):
                duplicated.append(futures[future])
    if duplicated:
        raise SystemExit(f"Train init states duplicate an official test state for {duplicated}; change split.seed")
    if crashed:
        raise SystemExit(f"{len(crashed)} tasks failed: {crashed}. Rerun the same command; finished tasks are skipped.")
    log.info(f"Done: {len(specs)} tasks in {split['train_init_dir']}")


if __name__ == "__main__":
    main()

"""Seeded train init states for the LIBERO-PRO split (spec §11.2), plus the overlap check against the official 50.

Same procedure as LIBERO-PRO's notebooks/generate_init_states.py (sample placements on reset, store the sim state
with no settling) but seeded: state i uses seed `split.seed + i`. Official init files are never touched.
Run from modules/VLS (uses the same hyperparameters.yaml):
    ../../envs/vls/bin/python -m residual.make_train_inits env.suite_name=libero_object_swap env.task_ids_filter=[0,1]
Reads object poses from the simulator: evaluation-side data-split analysis only, never a policy input.
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from residual.rollout import make_adapter

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


@hydra.main(config_path=".", config_name="hyperparameters", version_base="1.3")
def main(cfg) -> None:
    split = cfg.split
    env_overrides = OmegaConf.to_container(cfg.env, resolve=True)
    task_ids = env_overrides.pop("task_ids_filter")
    out_dir = Path(split.train_init_dir).resolve() / cfg.env.suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        libero_env = make_adapter(env_overrides, task_id)._env[0]
        env, official = libero_env._env, np.asarray(libero_env._init_states)
        states = []
        for i in range(split.num_train_inits):
            env.seed(split.seed + i)
            env.reset()
            states.append(env.get_sim_state())
        states = np.stack(states)
        report = overlap_report(object_poses(env, states), object_poses(env, official),
                                split.min_pos_dist_m, split.min_rot_dist_deg)
        report.update(task_id=task_id, task=libero_env.task, seed=split.seed)
        (out_dir / f"{libero_env.task}.overlap.json").write_text(json.dumps(report, indent=2))
        log.info(f"task {task_id} ({libero_env.task}): {json.dumps(report)}")
        if report["duplicates"]:
            raise SystemExit(f"{report['duplicates']} train init states duplicate an official test state; "
                             f"change split.seed")
        np.save(out_dir / f"{libero_env.task}.npy", states)
        libero_env.close()


if __name__ == "__main__":
    main()

# Checks that RPent's --task t --seed k gives the same libero_goal_task scene (BDDL, language, object poses) as our fixed-init launcher's init state k.
# Usage: (.venv-vls, VLS dir)   python scratch/2026-11-26_rpent_init_state_match.py ours  OUT_DIR
#        (.venv-rpent, RPent dir) python /abs/path/scratch/2026-11-26_rpent_init_state_match.py rpent OUT_DIR
#        (either venv)            python scratch/2026-11-26_rpent_init_state_match.py compare OUT_DIR
import json
import os
import sys

import numpy as np

SUITE = "libero_goal_task"
TASK_IDS = [0, 1]
INIT_IDS = [0, 1, 2]
POS_TOL_M = 1e-3
# ours: also run the launcher's num_steps_wait no-op settle steps (fixed_init_launcher.py:29-30) before recording
SETTLE = os.environ.get("SETTLE", "0") == "1"


def object_poses(raw_obs):
    """Robosuite '<obj>_pos' / '<obj>_quat' entries, world frame (m, xyzw)."""
    poses = {}
    for key, value in raw_obs.items():
        if key.endswith("_pos") or key.endswith("_quat"):
            if key.startswith("robot0_"):
                continue
            poses[key] = np.asarray(value, dtype=np.float64).tolist()
    return poses


def dump_ours(out_dir):
    sys.path.insert(0, os.getcwd())
    from core.env_adapters.libero_adapter import create_libero_envs

    records = []
    for task_id in TASK_IDS:
        env = create_libero_envs(SUITE, task_ids_filter=[task_id])[0]
        # Same order as scratch/2026-11-19_fixed_init_launcher.py env_reset_fixed_init: reset, then set_init_state(k)
        for init_id in INIT_IDS:
            env._env.seed(0)
            env._env.reset()
            raw_obs = env._env.set_init_state(env._init_states[init_id])
            if SETTLE:
                from core.env_adapters.libero_adapter import get_libero_dummy_action
                for _ in range(env.num_steps_wait):
                    raw_obs, _, _, _ = env._env.step(get_libero_dummy_action())
            records.append({
                "task_id": task_id,
                "init_id": init_id,
                "language": env.task_description,
                "bddl": env._env.env.bddl_file_name if hasattr(env._env, "env") else None,
                "n_init_states": len(env._init_states),
                "poses": object_poses(raw_obs),
            })
        env.close()
    return records


def dump_rpent(out_dir):
    sys.path.insert(0, os.getcwd())
    from robots.libero.env_server import make_env

    records = []
    for task_id in TASK_IDS:
        for seed in INIT_IDS:
            env = make_env(task_id=task_id, seed=seed, suite_name=SUITE)
            env.reset()
            raw_obs = env.current_raw_obs[0]
            records.append({
                "task_id": task_id,
                "init_id": seed,
                "language": env.task_descriptions[0],
                "bddl": None,
                "n_init_states": None,
                "poses": object_poses(raw_obs),
            })
            env.close()
    return records


def compare(out_dir):
    ours_file = "ours_settled.json" if SETTLE else "ours.json"
    print(f"comparing {ours_file} vs rpent.json")
    with open(os.path.join(out_dir, ours_file)) as f:
        ours = json.load(f)
    with open(os.path.join(out_dir, "rpent.json")) as f:
        rpent = json.load(f)
    rpent_by_key = {(r["task_id"], r["init_id"]): r for r in rpent}
    for o in ours:
        r = rpent_by_key[(o["task_id"], o["init_id"])]
        print(f"task {o['task_id']} init {o['init_id']}")
        print(f"  language ours : {o['language']!r}")
        print(f"  language rpent: {r['language']!r}  -> {'match' if o['language'] == r['language'] else 'DIFFERENT'}")
        only_ours = sorted(set(o["poses"]) - set(r["poses"]))
        only_rpent = sorted(set(r["poses"]) - set(o["poses"]))
        if only_ours or only_rpent:
            print(f"  keys only in ours: {only_ours}")
            print(f"  keys only in rpent: {only_rpent}")
        for key in sorted(set(o["poses"]) & set(r["poses"])):
            if not key.endswith("_pos"):
                continue
            a = np.array(o["poses"][key])
            b = np.array(r["poses"][key])
            dist_m = float(np.linalg.norm(a - b))
            verdict = "match" if dist_m < POS_TOL_M else "DIFFERENT"
            print(f"  {key:40s} ours {np.round(a, 4)}  rpent {np.round(b, 4)}  |d| {dist_m * 100:.2f} cm  {verdict}")


if __name__ == "__main__":
    mode, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if mode == "compare":
        compare(out_dir)
    else:
        records = dump_ours(out_dir) if mode == "ours" else dump_rpent(out_dir)
        suffix = "_settled" if (mode == "ours" and SETTLE) else ""
        with open(os.path.join(out_dir, f"{mode}{suffix}.json"), "w") as f:
            json.dump(records, f, indent=1)
        print(f"wrote {len(records)} records")

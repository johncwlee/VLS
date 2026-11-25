# Replays a logged task 0 episode open-loop from LIBERO init state 0 and logs gripper contacts and drawer joints per step.
import json
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import core.env_adapters.libero_adapter as libero_adapter

EPISODE_DIR = sys.argv[1]  # e.g. outputs/t0i0_ee_pose/2026-11-24_02-41-33/episode_1
OUT_PATH = os.path.join(sys.argv[2], 'replay_contacts.jsonl')
INIT_STATE = 0
SEED = 10_000  # scratch/2026-11-19_fixed_init_launcher.py: 10000 + episode index 0
MATCH_TOL_M = 1e-4

rows = [json.loads(line) for line in open(os.path.join(EPISODE_DIR, 'steps.jsonl'))]
env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[0])[0]

# Same order as the fixed-init launcher: reset (re-samples layout), set init state, settle with dummy actions
gym.Env.reset(env, seed=SEED)
env._env.seed(SEED)
env._env.reset()
env._env.set_init_state(env._init_states[INIT_STATE])
for _ in range(env.num_steps_wait):
    env._env.step(libero_adapter.get_libero_dummy_action())

rs_env = env._env.env  # robosuite / LIBERO problem env
sim = rs_env.sim
model = sim.model
controller = rs_env.robots[0].controller

gripper_geom_ids = set()
for gid in range(model.ngeom):
    body_name = model.body_id2name(model.geom_bodyid[gid])
    if 'gripper' in body_name or 'hand' in body_name or 'finger' in body_name or 'eef' in body_name:
        gripper_geom_ids.add(gid)
cabinet_joints = [model.joint_id2name(j) for j in range(model.njnt) if 'cabinet' in (model.joint_id2name(j) or '')]
print('gripper bodies:', sorted({model.body_id2name(model.geom_bodyid[g]) for g in gripper_geom_ids}))
print('cabinet joints:', cabinet_joints)


def geom_label(gid):
    name = model.geom_id2name(gid)
    return f'{model.body_id2name(model.geom_bodyid[gid])}/{name if name else gid}'


def contacts_now():
    """(gripper geom, other geom) labels for every active contact with the gripper; other is not the robot."""
    pairs = []
    for k in range(sim.data.ncon):
        c = sim.data.contact[k]
        a, b = c.geom1, c.geom2
        if a in gripper_geom_ids and b not in gripper_geom_ids:
            pairs.append((geom_label(a), geom_label(b)))
        elif b in gripper_geom_ids and a not in gripper_geom_ids:
            pairs.append((geom_label(b), geom_label(a)))
    return sorted({p for p in pairs if not p[1].startswith('robot0')})


ee_start = np.array(controller.ee_pos)
print(f'start EE: replay {np.round(ee_start, 5).tolist()} logged {np.round(rows[0]["ee_pos_before"], 5).tolist()} '
      f'err {np.linalg.norm(ee_start - rows[0]["ee_pos_before"]) * 1000:.3f} mm')

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
max_err_m = 0.0
with open(OUT_PATH, 'w') as f:
    for row in rows:
        action = np.array(row['action'], dtype=np.float64)
        action[-1] = 1.0 if action[-1] > 0 else -1.0  # same binarization as LiberoAdapter.step
        env._env.step(action)
        ee_pos_world_m = np.array(controller.ee_pos)
        err_m = float(np.linalg.norm(ee_pos_world_m - row['ee_pos_after']))
        max_err_m = max(max_err_m, err_m)
        drawer_qpos_m = {j: float(sim.data.qpos[model.get_joint_qpos_addr(j)]) for j in cabinet_joints}
        record = {'step': row['step'], 'ee_err_mm': 1000 * err_m, 'contacts': contacts_now(), 'drawer_qpos_m': drawer_qpos_m}
        f.write(json.dumps(record) + '\n')
print(f'replay vs log: max EE error {max_err_m * 1000:.3f} mm over {len(rows)} steps '
      f'-> {"match" if max_err_m < MATCH_TOL_M else "DIVERGED"} (tol {MATCH_TOL_M * 1000:.1f} mm)')
print(f'wrote {OUT_PATH}')

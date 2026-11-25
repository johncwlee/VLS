# Checks whether call 6's target pose (task 0, pose mode) penetrates the table/cabinet, and measures the gripper's extent in the EE frame.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import core.env_adapters.libero_adapter as libero_adapter

SITE = 'gripper0_grip_site'
OPEN_FINGER_QPOS = [0.0388, -0.0388]
# Call 6 target from scratch/2026-11-24_render_target_pose.py: kp13 + 1.1 cm Y, finger line -Z, approach -Y
TARGET_POS_M = np.array([0.0561, -0.1273, 0.9521])
TARGET_R = np.array([[0.0, 1.0, 0.0],   # columns: x = finger line (0, 0, -1), y = (1, 0, 0), z = approach (0, -1, 0)
                     [0.0, 0.0, -1.0],
                     [-1.0, 0.0, 0.0]])
GRIPPER_BODY_KEYS = ('gripper0', 'robot0_right_hand')


def matrix_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def place_ee_by_ik(sim, model, target_pos, target_R):
    site_id = model.site_name2id(SITE)
    for _ in range(500):
        sim.forward()
        err = np.concatenate([target_pos - sim.data.site_xpos[site_id],
                              matrix_to_axis_angle(target_R @ sim.data.site_xmat[site_id].reshape(3, 3).T)])
        J = np.concatenate([sim.data.get_site_jacp(SITE).reshape(3, -1), sim.data.get_site_jacr(SITE).reshape(3, -1)])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        sim.data.qpos[:7] = np.clip(sim.data.qpos[:7] + 0.5 * dq, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
    sim.data.qpos[7:9] = OPEN_FINGER_QPOS
    sim.forward()
    return np.linalg.norm(target_pos - sim.data.site_xpos[site_id])


def gripper_geom_ids(model):
    ids = []
    for gid in range(model.ngeom):
        body = model.body_id2name(model.geom_bodyid[gid])
        if any(key in body for key in GRIPPER_BODY_KEYS) and model.geom_contype[gid] + model.geom_conaffinity[gid] > 0:
            ids.append(gid)
    return ids


env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[0])[0]
env._env.reset()
env._env.set_init_state(env._init_states[0])
sim, model = env._env.env.sim, env._env.env.sim.model
raw_model = model._model  # mujoco.MjModel: geom_aabb is (ngeom, 6) = center (3), half sizes (3) in the geom frame

for gid in range(model.ngeom):
    name = model.geom_id2name(gid) or ''
    if 'table' in name and model.geom_type[gid] == 6:  # box
        print(f'table geom {name}: center z {sim.data.geom_xpos[gid][2]:.4f}, top z {sim.data.geom_xpos[gid][2] + model.geom_size[gid][2]:.4f} m')

residual_m = place_ee_by_ik(sim, model, TARGET_POS_M, TARGET_R)
print(f'IK residual {residual_m * 1000:.2f} mm')

# Gripper extent in the EE frame: corners of each collision geom's AABB, expressed relative to the grip site
site_id = model.site_name2id(SITE)
site_pos, site_R = sim.data.site_xpos[site_id], sim.data.site_xmat[site_id].reshape(3, 3)
corners_ee = []
lowest = (np.inf, None)
for gid in gripper_geom_ids(model):
    center, half = raw_model.geom_aabb[gid, :3], raw_model.geom_aabb[gid, 3:]
    geom_R = sim.data.geom_xmat[gid].reshape(3, 3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner_world = sim.data.geom_xpos[gid] + geom_R @ (center + half * np.array([sx, sy, sz]))
                corners_ee.append(site_R.T @ (corner_world - site_pos))
                if corner_world[2] < lowest[0]:
                    lowest = (corner_world[2], f'{model.body_id2name(model.geom_bodyid[gid])}/{model.geom_id2name(gid) or gid}')
corners_ee = np.array(corners_ee)  # (N, 3) EE frame: x finger line, y across, z approach (+z toward fingertips)
print('gripper collision-geom extent in EE frame (cm): '
      f'x (finger line) {100 * corners_ee[:, 0].min():.1f}..{100 * corners_ee[:, 0].max():.1f}, '
      f'y {100 * corners_ee[:, 1].min():.1f}..{100 * corners_ee[:, 1].max():.1f}, '
      f'z (approach) {100 * corners_ee[:, 2].min():.1f}..{100 * corners_ee[:, 2].max():.1f}')
print(f'lowest gripper AABB corner at target: z {lowest[0]:.4f} m ({lowest[1]})')

print('penetrating contacts at the target pose (dist < 0):')
grip_ids = set(gripper_geom_ids(model))
robot_ids = {g for g in range(model.ngeom) if model.body_id2name(model.geom_bodyid[g]).startswith(('robot0', 'gripper0'))}
for k in range(sim.data.ncon):
    c = sim.data.contact[k]
    if c.dist >= 0:
        continue
    a, b = c.geom1, c.geom2
    if a in robot_ids or b in robot_ids:
        robot_g, other_g = (a, b) if a in robot_ids else (b, a)
        if other_g in robot_ids:
            continue
        print(f'  {model.body_id2name(model.geom_bodyid[robot_g])}/{model.geom_id2name(robot_g) or robot_g} -> '
              f'{model.body_id2name(model.geom_bodyid[other_g])}/{model.geom_id2name(other_g) or other_g}: depth {-c.dist * 100:.1f} cm')

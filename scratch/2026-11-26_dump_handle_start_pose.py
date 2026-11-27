# Dumps our handle-start reset on libero_goal_task task 0, init states 0-2: the OSC waypoints it drives through and the eef pose it ends at, as robot0_eef_pos / robot0_eef_quat (xyzw), for replaying it in RPent's env.
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SCRATCH = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('handle_start_launcher', os.path.join(SCRATCH, '2026-11-25_handle_start_launcher.py'))
handle_start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handle_start)
libero_adapter = handle_start.libero_adapter
start_pose = handle_start.start_pose

waypoints = []
original_drive_osc = start_pose.drive_osc


def recording_drive_osc(env, target_world_m, R_target, max_steps, tol_m=0.005, tol_deg=1.0):
    robosuite_env = env._env.env
    # R_target is in the controller (grip site) frame; robot0_eef_quat is the hand body. Map with the fixed offset.
    R_obs_now = Rotation.from_quat(robosuite_env._get_observations()['robot0_eef_quat']).as_matrix()
    R_obs_to_ctrl = R_obs_now.T @ np.array(robosuite_env.robots[0].controller.ee_ori_mat)
    R_target_obs = np.asarray(R_target) @ R_obs_to_ctrl.T
    waypoints.append({'eef_pos': np.asarray(target_world_m).tolist(),
                      'eef_quat_xyzw': Rotation.from_matrix(R_target_obs).as_quat().tolist(),
                      'max_steps': max_steps, 'tol_m': tol_m, 'tol_deg': tol_deg})
    return original_drive_osc(env, target_world_m, R_target, max_steps, tol_m, tol_deg)


start_pose.drive_osc = recording_drive_osc

env = libero_adapter.create_libero_envs('libero_goal_task', task_ids_filter=[0])[0]
records = []
for init_index in [0, 1, 2]:
    handle_start.launcher.current['init_index'] = init_index
    waypoints = []
    env.reset()  # fixed init + settle + drive to the bottom handle + one hold step
    raw_obs = env._env.env._get_observations()
    records.append({'init_index': init_index,
                    'waypoints': waypoints,
                    'reached_eef_pos': np.asarray(raw_obs['robot0_eef_pos']).tolist(),
                    'reached_eef_quat_xyzw': np.asarray(raw_obs['robot0_eef_quat']).tolist()})
    print(json.dumps({k: records[-1][k] for k in ('init_index', 'reached_eef_pos')}), len(waypoints), 'waypoints')
with open(sys.argv[1], 'w') as f:
    json.dump(records, f, indent=1)

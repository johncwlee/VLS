# Fixed-init launcher, plus: after reset the gripper is driven (OSC) to the eef pose RPent's move_pose reached in t0_s{k}, and pi0.5 gets RPent's short prompt.
import importlib.util
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

SCRATCH = os.path.dirname(os.path.abspath(__file__))
# RPent's pi0_doubled prompt for libero_goal_task task 0 (outputs/rpent_goal_task_t01/2026-11-26_18-42-23)
SHORT_PROMPT = 'pull the lowest gray drawer handle outward'
# robot0_eef_pos (world, m) and robot0_eef_quat (xyzw) after RPent's move_pose, states.json step 1 of t0_s{k}.
# RPent's seed k = our init state k (scratch/2026-11-26_rpent_init_state_match.py).
RPENT_RUN = os.path.join(SCRATCH, '..', 'outputs', 'rpent_goal_task_t01', '2026-11-26_18-42-23')


def load_scratch_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRATCH, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_scratch_module('fixed_init_launcher', '2026-11-19_fixed_init_launcher.py')  # applies the fixed-init patches
start_pose = load_scratch_module('handle_start_pose', '2026-11-25_handle_start_pose.py')
libero_adapter = launcher.libero_adapter
fixed_init_reset = libero_adapter.LiberoEnv.reset


def rpent_start_pose(init_index):
    with open(os.path.join(RPENT_RUN, f't0_s{init_index}', 'states.json')) as f:
        step = json.load(f)['steps'][1]
    assert step['command']['action'] == 'move_pose', step['command']
    return np.array(step['state']['robot0_eef_pos']), np.array(step['state']['robot0_eef_quat'])


def reset_then_drive_to_rpent_pose(self, seed=None, **kwargs):
    if not hasattr(self, 'original_task_description'):
        self.original_task_description = self.task_description
    assert 'bottom drawer' in self.original_task_description, self.original_task_description
    fixed_init_reset(self, seed=seed, **kwargs)
    init_index = launcher.current['init_index']
    target_pos_world_m, target_quat_xyzw = rpent_start_pose(init_index)

    robosuite_env = self._env.env
    controller = robosuite_env.robots[0].controller
    # robot0_eef_quat is the hand body; controller.ee_ori_mat is the grip site (90 deg apart about the gripper z).
    # The offset between them is fixed, so measure it here and map RPent's quat into the controller frame.
    R_obs_now = Rotation.from_quat(robosuite_env._get_observations()['robot0_eef_quat']).as_matrix()
    R_obs_to_ctrl = R_obs_now.T @ np.array(controller.ee_ori_mat)
    R_target_ctrl = Rotation.from_quat(target_quat_xyzw).as_matrix() @ R_obs_to_ctrl
    contacts = start_pose.drive_osc(self, target_pos_world_m, R_target_ctrl, max_steps=300)
    raw_obs, _, _, _ = self._env.step(libero_adapter.get_libero_dummy_action())  # zero OSC delta: holds the pose

    reached_rotation_error_deg = start_pose.rotation_angle_deg(
        Rotation.from_quat(raw_obs['robot0_eef_quat']).as_matrix(), Rotation.from_quat(target_quat_xyzw).as_matrix())
    record = {'init_index': init_index,
              'target_eef_pos_world_m': np.round(target_pos_world_m, 4).tolist(),
              'reached_eef_pos_world_m': np.round(raw_obs['robot0_eef_pos'], 4).tolist(),
              'pos_error_cm': round(100 * float(np.linalg.norm(raw_obs['robot0_eef_pos'] - target_pos_world_m)), 2),
              'rot_error_deg': round(reached_rotation_error_deg, 2),
              'contacts_on_the_way': sorted(contacts),
              'contacts_at_start': sorted(start_pose.robot_contacts(robosuite_env.sim))}
    print(f'[rpent_start] {json.dumps(record)}', flush=True)
    self.task_description = SHORT_PROMPT  # read by the policy obs (libero_adapter.py:901-902)
    print(f'[short_prompt] {self.original_task_description!r} -> {SHORT_PROMPT!r}', flush=True)
    return self._format_raw_obs(raw_obs), {"is_success": False}


libero_adapter.LiberoEnv.reset = reset_then_drive_to_rpent_pose

if __name__ == '__main__':
    launcher.run()

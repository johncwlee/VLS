# Handle-start launcher with pi0.5's own pre-hook orientation (middle-drawer success, step 90) instead of the 30 deg-down pose.
import importlib.util
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

SCRATCH = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('handle_start_launcher', os.path.join(SCRATCH, '2026-11-25_handle_start_launcher.py'))
handle_launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handle_launcher)  # applies the fixed-init and handle-start patches
start_pose = handle_launcher.start_pose
libero_adapter = handle_launcher.libero_adapter

# pi0.5 alone, libero_goal task 0 init 0, success at step 115; step 90 is just before it pulls the drawer
VLA_RUN_STEPS = os.path.join(SCRATCH, '..', 'outputs', 'normal_start_vla_middle', '2026-11-26_00-06-19', 'episode_1',
                             'steps.jsonl')
VLA_STEP = 90


def load_vla_orientation():
    with open(VLA_RUN_STEPS) as f:
        rows = [json.loads(line) for line in f]
    row = [r for r in rows if r['step'] == VLA_STEP]
    assert len(row) == 1, f'step {VLA_STEP} not found once in {VLA_RUN_STEPS}'
    w, x, y, z = row[0]['ee_quat_wxyz_before']
    return Rotation.from_quat([x, y, z, w]).as_matrix()  # (3, 3) EE orientation, world frame


R_VLA_HOOK = load_vla_orientation()


def reset_then_drive_to_handle(self, seed=None, **kwargs):
    handle_launcher.fixed_init_reset(self, seed=seed, **kwargs)
    drawers = [d for d in handle_launcher.HANDLE_GEOM_BY_DRAWER if f'{d} drawer' in self.task_description]
    assert len(drawers) == 1, f'expected one drawer in the task text: {self.task_description!r}'
    start_pose.HANDLE_GEOM = handle_launcher.HANDLE_GEOM_BY_DRAWER[drawers[0]]
    error_m, rotation_deg, contacts, handle_pos, target = start_pose.drive_to_handle(self, None, R_VLA_HOOK)
    raw_obs, _, _, _ = self._env.step(libero_adapter.get_libero_dummy_action())  # zero OSC delta: holds the pose
    R_now = np.array(self._env.env.robots[0].controller.ee_ori_mat)
    record = {'init_index': handle_launcher.launcher.current['init_index'], 'handle_geom': start_pose.HANDLE_GEOM,
              'pose_error_cm': round(100 * error_m, 2), 'rotation_from_start_deg': round(rotation_deg, 1),
              'orientation_error_deg': round(start_pose.rotation_angle_deg(R_VLA_HOOK, R_now), 1),
              'gripper_z': np.round(R_now[:, 2], 2).tolist(), 'finger_axis': np.round(R_now[:, 0], 2).tolist(),
              'contacts_on_the_way': sorted(contacts),
              'contacts_at_start': sorted(start_pose.robot_contacts(self._env.env.sim)),
              'handle_center_world_m': np.round(handle_pos, 4).tolist(), 'target_world_m': np.round(target, 4).tolist()}
    print(f'[handle_start] {json.dumps(record)}', flush=True)
    return self._format_raw_obs(raw_obs), {"is_success": False}


libero_adapter.LiberoEnv.reset = reset_then_drive_to_handle

if __name__ == '__main__':
    handle_launcher.launcher.run()

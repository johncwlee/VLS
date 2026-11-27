# Fixed-init launcher, plus: after reset the gripper is driven (OSC, sim handle position) to 5 cm from the task's drawer handle (middle or bottom) at 30 deg down; then main.py runs from there.
import importlib.util
import json
import os

import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))


def load_scratch_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRATCH, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_scratch_module('fixed_init_launcher', '2026-11-19_fixed_init_launcher.py')  # applies the fixed-init patches
start_pose = load_scratch_module('handle_start_pose', '2026-11-25_handle_start_pose.py')
libero_adapter = launcher.libero_adapter

DOWN_DEG = 30  # approach angle below horizontal, picked by the user from the renders
HANDLE_GEOM_BY_DRAWER = {'middle': 'wooden_cabinet_1_g29', 'bottom': 'wooden_cabinet_1_g40'}  # handle bar geoms
fixed_init_reset = libero_adapter.LiberoEnv.reset


def reset_then_drive_to_handle(self, seed=None, **kwargs):
    fixed_init_reset(self, seed=seed, **kwargs)
    drawers = [d for d in HANDLE_GEOM_BY_DRAWER if f'{d} drawer' in self.task_description]
    assert len(drawers) == 1, f'expected one drawer in the task text: {self.task_description!r}'
    start_pose.HANDLE_GEOM = HANDLE_GEOM_BY_DRAWER[drawers[0]]
    error_m, rotation_deg, contacts, handle_pos, target = start_pose.drive_to_handle(self, DOWN_DEG)
    raw_obs, _, _, _ = self._env.step(libero_adapter.get_libero_dummy_action())  # zero OSC delta: holds the pose
    record = {'init_index': launcher.current['init_index'], 'handle_geom': start_pose.HANDLE_GEOM,
              'pose_error_cm': round(100 * error_m, 2),
              'rotation_from_start_deg': round(rotation_deg, 1), 'contacts_on_the_way': sorted(contacts),
              'contacts_at_start': sorted(start_pose.robot_contacts(self._env.env.sim)),
              'handle_center_world_m': np.round(handle_pos, 4).tolist(), 'target_world_m': np.round(target, 4).tolist()}
    print(f'[handle_start] {json.dumps(record)}', flush=True)
    return self._format_raw_obs(raw_obs), {"is_success": False}


libero_adapter.LiberoEnv.reset = reset_then_drive_to_handle

if __name__ == '__main__':
    launcher.run()

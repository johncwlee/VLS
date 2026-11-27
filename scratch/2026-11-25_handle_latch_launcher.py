# Fixed-init launcher, plus: steering latches off for the rest of the episode once the EE is within 5 cm of a cabinet_bottom keypoint (runtime patch, main.py unchanged).
import importlib.util
import json
import os

import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('fixed_init_launcher', os.path.join(SCRATCH, '2026-11-19_fixed_init_launcher.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)  # applies the fixed-init reset patches
main_module = launcher.main_module

HANDLE_OBJECT = 'wooden_cabinet_1_cabinet_bottom'  # task 0 target: bottom drawer
LATCH_DISTANCE_M = 0.05
latch = {'off': False, 'episode_dir': None}

original_start_closed_loop = main_module.Main._start_closed_loop
original_closed_loop_step = main_module.Main._closed_loop_step


def start_closed_loop_reset_latch(self, episode_dir, *args, **kwargs):
    latch['off'] = False
    latch['episode_dir'] = episode_dir
    return original_start_closed_loop(self, episode_dir, *args, **kwargs)


def closed_loop_step_with_latch(self, stage_state, global_step, action_chunk, keypoints, mask_ids):
    stage_state = original_closed_loop_step(self, stage_state, global_step, action_chunk, keypoints, mask_ids)
    handle_ids = [k for k, name in self.keypoint_id_to_object.items() if name == HANDLE_OBJECT]
    assert handle_ids, f'no keypoint on {HANDLE_OBJECT}: {self.keypoint_id_to_object}'
    keypoints_world_m = np.asarray(keypoints)  # (K, 3) tracked keypoints, world frame
    ee_pos_world_m, _ = self.adapter._measure_ee()  # (3,) OSC grip site, world frame
    distances_m = np.linalg.norm(keypoints_world_m[handle_ids] - ee_pos_world_m, axis=1)
    nearest_m = float(distances_m.min())
    just_latched = not latch['off'] and nearest_m < LATCH_DISTANCE_M
    if just_latched:
        latch['off'] = True
    closed_loop_use_guidance = bool(stage_state['use_guidance'])
    if latch['off']:
        stage_state['use_guidance'] = False
    record = {'step': global_step, 'handle_ids': handle_ids, 'nearest_handle_cm': round(100 * nearest_m, 2),
              'closed_loop_use_guidance': closed_loop_use_guidance, 'latched_off': latch['off'],
              'just_latched': just_latched}
    print(f'[latch] {json.dumps(record)}', flush=True)
    with open(os.path.join(latch['episode_dir'], 'latch.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')
    return stage_state


main_module.Main._start_closed_loop = start_closed_loop_reset_latch
main_module.Main._closed_loop_step = closed_loop_step_with_latch

if __name__ == '__main__':
    launcher.run()

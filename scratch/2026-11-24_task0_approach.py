# Task 0 episodes: closest EE approach to keypoint 13 / bottom handle, and finger-line angle to the handle bar there.
import json
import re
import sys

import numpy as np

# Bottom handle center (world, m) from sim geoms 201-203, init state 0 only (notes/handoff.md). Bar runs along world X.
HANDLE_CENTER_INIT0_M = np.array([0.037, -0.129, 0.946])
BAR_AXIS_WORLD = np.array([1.0, 0.0, 0.0])


def quat_wxyz_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def keypoint_from_prompt(prompt_path, index):
    text = open(prompt_path).read()
    block = text.split('Keypoint 3D positions [X, Y, Z] (current positions): "')[1].split('"')[0]
    rows = re.findall(r'\[\s*([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s*\]', block)
    return np.array([float(v) for v in rows[index]])


for episode_dir in sys.argv[1:]:
    import glob
    first_prompt = sorted(glob.glob(f'{episode_dir}/closed_loop/call_*/prompt.txt'))[0]
    kp13_world_m = keypoint_from_prompt(first_prompt, 13)
    steps = [json.loads(line) for line in open(f'{episode_dir}/steps.jsonl')]
    ee_pos_world_m = np.array([s['ee_pos_after'] for s in steps])  # (N, 3)
    dist_kp13_m = np.linalg.norm(ee_pos_world_m - kp13_world_m, axis=1)
    dist_handle_m = np.linalg.norm(ee_pos_world_m - HANDLE_CENTER_INIT0_M, axis=1)
    for name, dist in [('kp13', dist_kp13_m), ('handle_sim', dist_handle_m)]:
        i = int(np.argmin(dist))
        R_world_ee = quat_wxyz_to_matrix(np.array(steps[i]['ee_quat_wxyz_after']))
        finger_line_world = R_world_ee[:, 0]  # EE x = finger line (LIBERO Panda)
        approach_world = R_world_ee[:, 2]
        angle_to_bar_deg = np.degrees(np.arccos(abs(finger_line_world @ BAR_AXIS_WORLD)))  # 90 = across the bar
        width_cm = 100 * (steps[i]['finger_qpos_m'][0] - steps[i]['finger_qpos_m'][1])
        print(f'{episode_dir} {name}: min {100 * dist[i]:.1f} cm at step {steps[i]["step"]}, '
              f'finger line {np.round(finger_line_world, 2)}, angle to bar {angle_to_bar_deg:.0f} deg, '
              f'approach {np.round(approach_world, 2)}, width {width_cm:.1f} cm')
    print(f'{episode_dir} kp13 {np.round(kp13_world_m, 3)}, steps {len(steps)}, '
          f'any success {any(s["success"] for s in steps)}')
